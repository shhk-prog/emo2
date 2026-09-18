# 全再実行・リポジトリ公開に向けた最終改修の確認 (Walkthrough)

## 1. 概要
本改修では、全再実行および公開に向けた最終的な不整合・課題の解消を行い、リポジトリ全体の一貫性、再現性、厳密性を確立しました。

---

## 2. 実施した改修内容の詳細

### 2.1 統合 CLI (`--stage v1`) の完全パイプライン fan-out
- **改修前**: `v1/primary/phase_c/summarize_phase_c.py` のみを呼び出しており、Phase A〜C の実験本体が実行されていませんでした。
- **改修後**: `src/affective_empathy_eval/run.py` の `run_v1()` を改修し、`configs/models.yaml` の全モデル（Base / Instruct）に対して：
  1. Phase A: `v1/primary/run_phase_a.py` (E1 Decodability & E2 Geometry)
  2. Phase B: `v1/primary/run_phase_b.py` (Semantic Controls Audit)
  3. Phase C: `v1/primary/run_phase_c.py` (E3 Causal Map & E4 Interchangeability)
  4. Phase C E6: `v1/primary/phase_c/run_e6_specialization.py` (Targeted Ablation & LMM)
  を順次 fan-out 実行し、最後に `summarize_phase_c.py` によるクロスモデル要約を生成する完全自動化パイプラインを確立しました。

### 2.2 統合 CLI (`--stage behavioral`) の完全網羅化
- **改修前**: `run_behavioral_emobank.py` を Instruct モデルのみ実行していました。
- **改修後**: `run.py` の `run_behavioral()` において、全モデル（Base / Instruct）$\times$ 2大データセット（EmoBank 3-Way VAD + AIPsy-Affect 4-Split）を網羅実行するように修正しました。

### 2.3 V1 Primary スクリプト群の ModelRegistry 完全対応
- `v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`, `v1/primary/phase_c/run_e6_specialization.py` において、`add_model_selection_args`, `resolve_models_from_args` を適用。
- コマンドライン引数 (`--model-set`, `--family`, `--base-model`, `--instruct-model`) からの動的解決に対応しました。
- 各スクリプトに `--dry-run` モードを実装し、軽量かつ決定論的なパイプライン検証を可能にしました。

### 2.4 V1 Phase B の Layer 14 固定廃止と相対深さ動的解決
- **改修前**: Qwen 28層前提の `--layer 14` がハードコードされていました。
- **改修後**: `--layer None`（未指定時動的計算）および `--relative-depth 0.5` を導入。`resolve_architecture_dims(model_id)` またはモデル config の総層数 $L$ から、`target_layer = int(round(relative_depth * (L - 1)))` によりモデルごとの同等深度層を自動算出するように改修しました。

### 2.5 `configs/v3_experiments.yaml` の具体的モデル ID 完全排除
- `target_model: "Qwen/Qwen2.5-1.5B-Instruct"` $\rightarrow$ `target_family: "qwen"`
- `confirmatory_models: [...]` $\rightarrow$ `confirmatory_families: ["llama", "gemma", "olmo"]`
- V3 スクリプト群（`run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py`）が `configs/models.yaml` を介して動的にモデル ID とアーキテクチャパラメータを解決する構成へ移行しました。

### 2.6 `scripts/run_scale_validation.py` の疑似ダミー値撤廃と V2 Primary 正式ラッパー化
- **改修前**: ハードコードされた固定ダミー JSON 値を出力するだけのプレースホルダーでした。
- **改修後**: V2 Primary スクリプト（RQ1/RQ2, RQ3, RQ4）を `--models-config configs/models.yaml --model-set scale_validation` で実際にサブプロセス実行する正式なオーケストレーションラッパーへ改修しました。

### 2.7 旧スクリプト群・旧論文ドラフトの `legacy/` 完全隔離
- `v1/scripts/` 内の過渡期スクリプト群（59ファイル）を `v1/scripts/legacy/` 配下へ隔離。
- `v3/docs/` 内の旧論文ドラフト（`paper.md`, `paper2.md`, `paper3.md`, `paper_v1.md`, `paper_restructured.md`, `Related Work.md`, `emo v1–v3 の新規性評価.md`）を `v3/docs/legacy/` 配下へ隔離。
- `v3/scripts/` 内の過渡期スクリプト群（27ファイル）を `v3/scripts/legacy/` 配下へ隔離。
- それぞれのディレクトリに正本実装への参照を記した `README.md` を配備しました。

### 2.8 レジストリ安全機構の強化 (`src/affective_empathy_eval/models/registry.py`)
- `resolve_architecture_dims`: 未知モデルに対して Qwen (28, 1536) へ暗黙フォールバックする危険な挙動を廃止し、明確な `ValueError` を送出。
- `resolve_models_from_args`: `--base-model` または `--instruct-model` を指定した際、`--family` が指定されていない場合は `ValueError` を送出して誤爆を防止。

### 2.9 V3 Path Mediation & RQ1 の概念・用語・統計の厳密化
- **NDE/NIE 用語の完全撤廃**:
  - `total_affective_shift`: 自然な情動変位の総量
  - `residual_shift_after_blocking`: Mediator 層遮断後の残差変位
  - `mediated_attenuation`: Mediator 層遮断による情動変位の減衰量
  - `attenuation_ratio`: 減衰率 (`mediated_attenuation / total_affective_shift`)
  JSON キーおよびコード内変数も含めて NDE/NIE を完全排除しました。
- **Discovery プローブの GroupKFold 化**:
  - `GroupKFold(n_splits=min(5, n_groups))` により、同一 stimulus pair のデータ漏洩（leakage）を完全に排除した CV $R^2$ 評価を実装しました。
- **RQ1 Topic Control の位置づけの明確化**:
  - 介入が汎用的な表現破壊ではなく情動自己報告に特異的であることを検証する「非特異的除外統制（non-specific exclusion control）」である旨をコード・ログ・docstring に明記しました。

### 2.10 RunManifest の標準化実装 (`src/affective_empathy_eval/manifests.py`)
- `RunManifest` データクラスおよび `create_run_manifest` ヘルパー関数を実装し、Git SHA、UTC タイムスタンプ、実験設定、メタデータを一貫して JSON 保管できるようにしました。

### 2.11 キャッシュ削除・成果物ディレクトリのクリーン初期化
- 全 `__pycache__` ディレクトリおよび `*.pyc` ファイルを完全削除。
- `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/`, `results/ablation/` 配下の古い実行成果物をクリーン初期化（`.gitkeep` のみ保持）。

---

## 3. 検証結果

### 3.1 単体テスト (`pytest`)
```bash
.venv/bin/pytest -q
```
- **結果**: **46 passed, 1 warning in 15.07s** (全テスト成功)

### 3.2 統合 CLI 各ステージの dry-run スモークテスト
| ステージ | 実行コマンド | 結果 | 確認事項 |
|---|---|---|---|
| **V1 Stage** | `python -m affective_empathy_eval.run --stage v1 --model-set primary_small --dry-run --family qwen` | **SUCCESS (exit 0)** | Base/Instruct 両モデルに対し Phase A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ E6 $\rightarrow$ summarize が完全完走 |
| **Behavioral** | `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --dry-run --family qwen` | **SUCCESS (exit 0)** | Base/Instruct 両モデルに対し EmoBank + AIPsy の両方が実行 |
| **V2 Stage** | `python -m affective_empathy_eval.run --stage v2 --model-set primary_small --dry-run --family qwen` | **SUCCESS (exit 0)** | RQ1/RQ2 Cross-decoding $\rightarrow$ RQ3 Causal Map $\rightarrow$ RQ4 Recovery Patching が完全完走 |
| **V3 Stage** | `python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run --family qwen` | **SUCCESS (exit 0)** | RQ1 Gate (GO) $\rightarrow$ RQ2 4-Maps $\rightarrow$ RQ3 Path Mediation (Attenuation Ratio: 0.738) $\rightarrow$ Step 7 Confirmatory (Llama, Gemma, OLMo) が完全完走 |
| **Scale Validation** | `python -m affective_empathy_eval.run --stage scale_validation --dry-run` | **SUCCESS (exit 0)** | Mistral 7B (32 layers) に対し V2 Primary スクリプト群が連携実行 |

---

## 4. 結論
全10項目および関連する基盤機構の改修が完了し、リポジトリは全実験の再実行および外部公開に耐えうる極めて高い完成度と再現性に達しました。
