# 共通モデルレジストリ刷新・1-1.5B Primary コホート統一・統合 CLI 完了報告 (Walkthrough)

## 概要
研究の中心課題である「事後学習（Post-training）による内部表現幾何・因果回路の再編」において、モデルサイズ交絡（Mistral 7B のみが突出して大規模であること）を完全に排除するため、Primary コホートを **1〜1.5B パラメータ帯（Qwen 1.5B, Llama 1B, Gemma 3 1B, OLMo 2 1B）** に統一しました。
また、Mistral 7B は「大規模モデルでの頑健性・再現性検証（Supplementary Scale Validation）」に移行しました。

さらに、コードベース内のあらゆるモデル ID・層数・隠れ層次元のハードコードを完全撤廃し、`configs/models.yaml` を唯一の正本として、共通パッケージ `affective_empathy_eval` 経由で動的解決・自動アダプテーション・統合実行できる仕組みを構築しました。

---

## 主な改修点と成果物

### 1. `configs/models.yaml` の再編
- **`model_sets:` 階層構造の導入**:
  - `primary_small`:
    - **Qwen**: `Qwen/Qwen2.5-1.5B` $\leftrightarrow$ `Qwen/Qwen2.5-1.5B-Instruct` (1.5B)
    - **Llama**: `meta-llama/Llama-3.2-1B` $\leftrightarrow$ `meta-llama/Llama-3.2-1B-Instruct` (1.23B)
    - **Gemma**: `google/gemma-3-1b-pt` $\leftrightarrow$ `google/gemma-3-1b-it` (1B, Gemma 3)
    - **OLMo**: `allenai/OLMo-2-0425-1B` $\leftrightarrow$ `allenai/OLMo-2-0425-1B-Instruct` (1B, OLMo 2)
  - `scale_validation`:
    - **Mistral**: `mistralai/Mistral-7B-v0.3` $\leftrightarrow$ `mistralai/Mistral-7B-Instruct-v0.3` (7B)
- **手書き層数・次元の撤廃**:
  - `num_layers`, `hidden_dim` の記述を YAML から排除し、HF `AutoConfig` からの自動取得に移行。

### 2. 共通 ModelRegistry の刷新 (`src/affective_empathy_eval/models/registry.py`)
- **`load_model_set` 関数**:
  - `load_model_set(config_path, model_set="primary_small")` により、指定されたコホートの全モデル定義を辞書形式で取得。
- **動的次元解決 (`resolve_architecture_dims`)**:
  - HuggingFace `AutoConfig.from_pretrained()` から `num_hidden_layers` および `hidden_size` を自動抽出。
  - オフライン実行時・テスト実行時・HF アクセス不可時のための既知モデルフォールバック辞書を保持し、環境に左右されない高信頼性を担保。
- **CLI 引数共通パーサー (`add_model_selection_args`, `resolve_models_from_args`)**:
  - `--model-set` (default: `primary_small`)
  - `--family` (特定ファミリーの絞り込み)
  - `--base-model`, `--instruct-model` (CLI による動的オーバーライド)
  - 優先順位: CLI > YAML > default を厳格に適用。

### 3. ModelAdapter の拡張 (`src/affective_empathy_eval/models/adapters.py`)
- **Gemma 3 対応 (`GemmaAdapter`)**: `Gemma3ForCausalLM` / `Gemma2ForCausalLM` 両対応。
- **OLMo 2 対応 (`Olmo2Adapter`)**: `OLMo2ForCausalLM`（Rotary Embedding, SwiGLU, RMSNorm）対応。
- **直接ディスパッチ**: `get_model_adapter(model, adapter_name="olmo2")` による明示的アダプター選択をサポート。

### 4. 各 Stage スクリプトにおけるハードコード撤廃
- **V2 Stage (`v2/primary/`)**:
  - `run_rq1_rq2_cross_decoding.py`: `--model-set` / `--family` 引数対応、動的モデル解決
  - `run_rq3_causal_map.py`: 動的モデル解決、`out_path.parent.mkdir()` 安全化
  - `run_rq4_recovery_patching.py`: 動的モデル解決、`out_path.parent.mkdir()` 安全化
- **V3 Stage (`v3/primary/`)**:
  - `run_rq1_state_induction.py`: 動的ターゲットモデル解決
  - `run_rq2_spatiotemporal_maps.py`: 動的層数取得（28層固定値の撤廃）
  - `run_rq3_path_mediation.py`: 動的層数取得、動的モデル解決
  - `run_confirmatory_replication.py`: OLMo 2 を含む Primary 3 モデルへの更新、動的層数取得
- **V1 Stage (`v1/primary/phase_c/summarize_phase_c.py`)**:
  - `get_models_from_config` を `load_model_set` 連携に改修し、モデル名ハードコードを全廃。

### 5. 統合 CLI エントリポイントの新設 (`src/affective_empathy_eval/run.py`)
ワンコマンドで各ステージの全実験を実行できる CLI を実装しました：
```bash
# Primary 1-1.5B コホートで各ステージを実行
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
python -m affective_empathy_eval.run --stage v1 --model-set primary_small
python -m affective_empathy_eval.run --stage v2 --model-set primary_small
python -m affective_empathy_eval.run --stage v3 --model-set primary_small

# Supplementary 7B 外部スケール検証 (Mistral 7B)
python -m affective_empathy_eval.run --stage v2 --model-set scale_validation

# Dry-run による高速動作検証
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run
```

---

## 検証結果

### 1. 全自動テストスイート (`.venv/bin/pytest -q`)
```text
.............................................                                               [100%]
45 passed in 15.17s
```
- 新規単体テスト `tests/test_model_registry_and_adapters.py`（5件）を含む全45件が完全合格。

### 2. 統合 CLI によるドライランスモークテスト
- **V3 ステージ (Primary 1-1.5B Cohort)**:
  - `python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run`
  - RQ1, RQ2, RQ3, Confirmatory (Llama, Gemma, OLMo) の全4プロセスが正常終了 (Code 0)。
- **V2 ステージ (Primary 1-1.5B Cohort)**:
  - `python -m affective_empathy_eval.run --stage v2 --model-set primary_small --dry-run --max-samples 100`
  - RQ1 & RQ2 クロスデコード、RQ3 因果マップ、RQ4 分布回復パッチングの全プロセスが Qwen, Llama, Gemma, OLMo の4ファミリーで正常終了 (Code 0)。
- **V2 ステージ (Supplementary 7B Cohort)**:
  - `python -m affective_empathy_eval.run --stage v2 --model-set scale_validation --dry-run --max-samples 50`
  - Mistral 7B v0.3 の全プロセスが正常終了 (Code 0)。

---

## 結論
本改修により、研究のコアである Primary 4 大モデルファミリーが 1〜1.5B 帯に美しく統制され、外部スケール検証（Mistral 7B）との役割分担が明確化されました。コードベース内のハードコードも完全に一掃され、ワンコマンドで全 Stage を自在に再現・スケール実行できる理想的な実験環境が完成しました。
