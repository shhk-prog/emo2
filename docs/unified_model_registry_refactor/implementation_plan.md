# 共通モデルレジストリ刷新・1-1.5B Primary コホート統一・統合 CLI 実装計画

## 概要
研究の中心である「Base $\rightarrow$ Instruct 事後学習による内部表現・因果利用の幾何学的再編」において、モデルサイズ交絡を排除し、査読者からの「7B だけモデル規模が異なることによるアーティファクトではないか」という懸念を解消するため、Primary 4 大モデルファミリーを **1〜1.5B 帯** に統一します。
また、Mistral 7B は「大規模モデルでの頑健性・再現性検証（External-Scale Robustness Replication）」として Supplementary に位置づけます。

さらに、コードベース内のあらゆるモデル ID・層数・隠れ層次元のハードコードを完全撤廃し、`configs/models.yaml` を唯一の正本として、共通パッケージ `affective_empathy_eval` 経由で動的解決・自動アダプテーション・統合実行できる仕組みを構築します。

---

## ユーザーレビュー必須項目

> [!IMPORTANT]
> **Primary コホートの更新と規模統一**:
> - **Primary 4-Family Cohort (`primary_small`, 1〜1.5B 帯)**:
>   1. **Qwen**: `Qwen/Qwen2.5-1.5B` $\leftrightarrow$ `Qwen/Qwen2.5-1.5B-Instruct` (1.5B)
>   2. **Llama**: `meta-llama/Llama-3.2-1B` $\leftrightarrow$ `meta-llama/Llama-3.2-1B-Instruct` (1.23B)
>   3. **Gemma**: `google/gemma-3-1b-pt` $\leftrightarrow$ `google/gemma-3-1b-it` (1B, Gemma 3 公式 Pretrained/IT)
>   4. **OLMo**: `allenai/OLMo-2-0425-1B` $\leftrightarrow$ `allenai/OLMo-2-0425-1B-Instruct` (1B, OLMo 2 公式 Base/SFT+DPO+RLVR)
> - **Supplementary Cohort (`scale_validation`, 7B 帯外部スケール検証)**:
>   - **Mistral**: `mistralai/Mistral-7B-v0.3` $\leftrightarrow$ `mistralai/Mistral-7B-Instruct-v0.3` (7B)

> [!NOTE]
> **互換性とオフラインフォールバック**:
> - 層数（`num_layers`）および隠れ層次元（`hidden_dim`）の手書き記述を YAML から排除し、HF の `AutoConfig`（`num_hidden_layers`, `hidden_size`）から動的取得します。
> - ただし、インターネット非接続環境でのテスト実行や dry-run 時にブロックされないよう、既知モデルに対するローカルメタデータフォールバックを保持し、どのような実行環境でも安定動作させます。

---

## 提案される変更点

### 1. 設定層 (`configs/`)

#### [MODIFY] [models.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/models.yaml)
- `model_sets:` 構造を導入：
  - `primary_small`: Qwen 1.5B, Llama 1B, Gemma 3 1B, OLMo 2 1B
  - `scale_validation`: Mistral 7B v0.3
- 各モデルに `adapter` (`qwen`, `llama`, `gemma3`, `olmo2`, `mistral`)、`scale`、`enabled: true` を付与。
- ハードコードされた `num_layers`, `hidden_dim` を削除。

---

### 2. 共通モデルパッケージ (`src/affective_empathy_eval/models/`)

#### [MODIFY] [registry.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/registry.py)
- `load_model_set(config_path="configs/models.yaml", model_set="primary_small") -> Dict[str, ModelFamilyConfig]` 関数の実装。
- `ModelFamilyConfig`:
  - `num_layers` / `hidden_dim` のプロパティ動的取得（HF `AutoConfig` 参照、オフライン用フォールバック辞書付き）。
  - `adapter_name: str` を保持。
- CLI 引数パーサーヘルパーの提供:
  - `--model-set` (デフォルト: `primary_small`)
  - `--family` (特定ファミリーの絞り込み: `qwen`, `llama`, `gemma`, `olmo`, `mistral`)
  - `--base-model` / `--instruct-model` (CLI による動的オーバーライド)
  - 優先順位: CLI > YAML > default
- `get_registry()` の後方互換性を完全維持。

#### [MODIFY] [adapters.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/adapters.py)
- `Gemma3Adapter` / `GemmaAdapter`: Gemma 3 (`Gemma3ForCausalLM` / `model.layers`) に対応。
- `Olmo2Adapter`: OLMo 2 (`OLMo2ForCausalLM` / `model.layers` / `transformer.blocks`) に対応。
- `get_model_adapter(model, adapter_name=None)`:
  - `adapter_name`（例: `"olmo2"`, `"gemma3"`, `"llama"`, `"qwen"`）指定およびクラス名からの柔軟な自動ディスパッチ。

---

### 3. 各 Stage スクリプトのモデル参照一本化 (`behavioral/`, `v1/`, `v2/`, `v3/`)

- コード内の `MODEL_IDS = [...]` やハードコードされたモデル辞書をすべて撤廃。
- すべて `registry.load_model_set()` または `resolve_models_from_args()` 経由で取得。
- 対象スクリプト:
  - `v1/primary/phase_c/summarize_phase_c.py`
  - `v2/primary/run_rq1_rq2_cross_decoding.py`
  - `v2/primary/run_rq3_causal_map.py`
  - `v2/primary/run_rq4_recovery_patching.py`
  - `v3/primary/run_rq1_state_induction.py`
  - `v3/primary/run_rq2_spatiotemporal_maps.py`
  - `v3/primary/run_rq3_path_mediation.py`
  - `v3/primary/run_confirmatory_replication.py`
  - `behavioral/` 配下スクリプト

---

### 4. 統合 CLI エントリポイント

#### [NEW] [src/affective_empathy_eval/run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `python -m affective_empathy_eval.run` から各ステージを統合実行できる CLI を提供：
  ```bash
  python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
  python -m affective_empathy_eval.run --stage v1 --model-set primary_small
  python -m affective_empathy_eval.run --stage v2 --model-set primary_small
  python -m affective_empathy_eval.run --stage v3 --model-set primary_small
  ```
- 外部スケール検証用：
  ```bash
  python -m affective_empathy_eval.run --stage v2 --model-set scale_validation
  ```
- `--dry-run`, `--family`, `--device` などの共通オプションを透過的に各ステージに伝達。

---

### 5. ドキュメント整備

#### [MODIFY] [README.md](file:///mnt/nas/home/hiromi/src/emo2/README.md)
#### [MODIFY] [v1/README.md](file:///mnt/nas/home/hiromi/src/emo2/v1/README.md)
#### [MODIFY] [v2/README.md](file:///mnt/nas/home/hiromi/src/emo2/v2/README.md)
#### [MODIFY] [v3/README.md](file:///mnt/nas/home/hiromi/src/emo2/v3/README.md)
- Primary コホート（1〜1.5B: Qwen, Llama, Gemma 3, OLMo 2）および Supplementary（Mistral 7B）の理論的位置づけを明記。
- 統合 CLI コマンドの Quick Start を記載。

---

## 検証計画

### 1. 自動テスト
- 新規単体テスト `tests/test_model_registry_and_adapters.py`:
  - `load_model_set("primary_small")` の読み込み検証
  - `load_model_set("scale_validation")` の読み込み検証
  - CLI 引数パースおよび override 機能の検証（CLI > YAML > default）
  - `resolve_architecture_dims` の動作検証（HF config / フォールバック辞書）
  - 各 Adapter (`qwen`, `llama`, `gemma3`, `olmo2`, `mistral`) のフックターゲット解決テスト
- 既存テストスイートの全件パス確認:
  - `.venv/bin/pytest -q`

### 2. ドライランスモークテスト
- 統合 CLI による各ステージの dry-run 実行確認:
  ```bash
  .venv/bin/python -m affective_empathy_eval.run --stage v2 --model-set primary_small --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run
  .venv/bin/python -m affective_empathy_eval.run --stage v2 --model-set scale_validation --dry-run
  ```
