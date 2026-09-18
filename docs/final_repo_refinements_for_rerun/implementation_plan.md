# 全再実行・公開に向けた最終完全化 実装計画

## 概要
全再実行（GPU本実行）およびオープンソース公開に向けて、コードベース全体の整合性・完全性を極限まで高めるため、以下の重要リファインメントを実施します：
1. 統合 CLI における V1（Phase A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ E6 $\rightarrow$ summary）および Behavioral（EmoBank + AIPsy、Base + Instruct）の完全自動 fan-out 実行
2. V1 Primary コード群の完全モデルレジストリ対応と、Phase B における層固定（Layer 14）の撤廃（relative depth $d \approx 0.5$ / peak からの動的解決）
3. `configs/v3_experiments.yaml` からの具体的モデル ID 完全撤廃（`target_family: qwen`, `confirmatory_families: [...]` によるレジストリ動的解決）
4. `scripts/run_scale_validation.py` の固定ダミー値撤廃と、V2 Primary を `--model-set scale_validation` で実際に走らせる正式ラッパー化
5. `KNOWN_MODEL_DIMS` における未知モデルの Qwen (28, 1536) 暗黙フォールバックの撤廃（`ValueError` 送出）
6. `--base-model` / `--instruct-model` CLI override 時の `--family` 指定必須化
7. V3 Path Mediation の NDE/NIE 用語完全撤廃、Discovery プローブの `GroupKFold(pair_id)` 化、RQ1 Topic Control の非特異的除外統制としての位置づけ整理
8. 旧 V1/V3 scripts および旧 neutralization / collapse ドキュメントの `legacy/` 隔離
9. `__pycache__` / `.pyc` の完全削除および `results/` のクリーン初期化

---

## ユーザーレビュー必須項目

> [!IMPORTANT]
> **V1 および Behavioral の完全 fan-out 実行**:
> - `python -m affective_empathy_eval.run --stage v1 --model-set primary_small`
>   - 4 大ファミリー（Qwen, Llama, Gemma, OLMo）× 2 水準（Base, Instruct）の全 8 モデルに対し、Phase A $\rightarrow$ Phase B $\rightarrow$ Phase C (E3/E4) $\rightarrow$ E6 $\rightarrow$ summarize を順次実行します。
> - `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small`
>   - 全 8 モデルに対し、EmoBank 3-Way VAD および AIPsy 4-Split の両方を実行します。

> [!IMPORTANT]
> **Scale Validation の実実行化**:
> - `scripts/run_scale_validation.py` 内に存在していた固定疑似結果（`delta_sharing_peak: 0.62` 等）を完全撤廃し、V2 Primary の RQ1〜RQ4 スイートを Mistral 7B に対して実際に実行・出力するラッパーに改修します。

---

## 提案される変更点

### 1. 共通モデルレジストリ安全強化 (`src/affective_empathy_eval/models/registry.py`)
- `resolve_architecture_dims(model_id)`: 未知モデルで AutoConfig もフォールバック辞書もヒットしない場合、暗黙の (28, 1536) ではなく `raise ValueError(...)` を送出。
- `resolve_models_from_args(args)`: `--base-model` または `--instruct-model` を指定する場合、`--family` の指定を必須化（複数モデルの誤爆防止）。

### 2. V1 Primary の完全動的化 (`v1/primary/`)
- `run_phase_a.py`, `phase_c/run_phase_c.py`, `run_e6_causal_specialization.py`: `add_model_selection_args` / `resolve_models_from_args` を導入し、ハードコードデフォルトを全廃。
- `run_phase_b.py`:
  - `--layer 14` 固定を廃止し、`--relative-depth 0.5`（デフォルト）から `target_layer = int(round(args.relative_depth * (num_layers - 1)))` を算出。

### 3. 統合ランナーの完全パイプライン化 (`src/affective_empathy_eval/run.py`)
- `run_v1()`: 全モデルに対する Phase A $\rightarrow$ Phase B $\rightarrow$ Phase C $\rightarrow$ E6 $\rightarrow$ summarize の実行ループを実装。
- `run_behavioral()`: 全モデル（Base/Instruct）に対する EmoBank + AIPsy の実行ループを実装。

### 4. V3 設定のクリーン化 (`configs/v3_experiments.yaml` & `v3/primary/`)
- `v3_experiments.yaml`:
  - `target_family: "qwen"`
  - `confirmatory_families: ["llama", "gemma", "olmo"]`
  - 具体的モデル ID ハードコードを完全削除。
- `run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py`: 設定ファイルの family 名からレジストリ経由で完全動的解決。

### 5. Scale Validation ラッパー化 (`scripts/run_scale_validation.py`)
- 固定疑似結果を撤廃。
- V2 Primary スクリプト（RQ1〜RQ4）を `--models-config configs/models.yaml --model-set scale_validation` で呼び出し、結果を `results/ablation/scale_validation/` 配下に保存する純粋なラッパーに改修。

### 6. V3 Path Mediation & RQ1 の理論的整合化
- `run_rq3_path_mediation.py`:
  - 残存 NDE/NIE 表記を `total affective shift`, `residual shift`, `mediated attenuation` へ完全置換。
  - Discovery プローブの分割を `GroupKFold(n_splits=5)` にし、`groups=discovery_df["pair_id"]` を明示。
- `run_rq1_state_induction.py`:
  - Self affective shift と Topic TVD を別個の指標として記録・表示し、非特異的摂動の除外統制であることを docstring/ログに明記。

### 7. 旧 scripts・docs の legacy 隔離
- `v1/scripts/legacy/`
- `v3/docs/legacy/` (`paper_restructured.md`, `paper2.md` 等)
- `v3/scripts/legacy/` (`plot_paper_figures.py` 等)
への移動。

### 8. キャッシュ削除および results 初期化
- `__pycache__` および `*.pyc` を全削除。
- 実行前クリーン状態にするため `results/` 配下の一時出力を整理。

---

## 検証計画

### 1. 単体テスト & 回帰テスト
- `tests/test_model_registry_and_adapters.py`: 未知モデル時の ValueError 送出、`--family` なしの override 時の ValueError 送出のテストを追加。
- `.venv/bin/pytest -q` の実行（全件合格確認）。

### 2. ドライランスモークテスト
- 統合ランナーによる全ステージのドライラン実行：
  ```bash
  .venv/bin/python -m affective_empathy_eval.run --stage v1 --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage behavioral --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage v2 --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage v3 --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage scale_validation --dry-run
  ```
