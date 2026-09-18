# タスクリスト: 共通モデルレジストリ刷新・1-1.5B Primary コホート統一・CLI 統合

## 目的
Primary モデルファミリーを 1〜1.5B 帯（Qwen 1.5B, Llama 1B, Gemma 3 1B, OLMo 2 1B）に統一し、Mistral 7B を外部スケール検証（Supplementary）へ移行する。モデル定義を `configs/models.yaml` に一本化し、コード内のハードコード（`MODEL_IDS`、層数、隠れ層次元など）を完全排除した上で、`python -m affective_empathy_eval.run` による統合実行を可能にする。

---

## タスク一覧

- [x] 1. **ドキュメント準備**:
  - [x] `docs/unified_model_registry_refactor/task.md` 作成
  - [x] `docs/unified_model_registry_refactor/implementation_plan.md` 作成・承認
- [x] 2. **`configs/models.yaml` の再編**:
  - [x] `model_sets:` 構造の導入（`primary_small`, `scale_validation`）
  - [x] 4大ファミリーの 1〜1.5B 化（Qwen 2.5 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B）
  - [x] Mistral 7B v0.3 を `scale_validation` に配置
  - [x] 手書きの層数・隠れ層次元記述の削除（HF Config から自動取得可能に）
- [x] 3. **共通 ModelRegistry の刷新 (`src/affective_empathy_eval/models/registry.py`)**:
  - [x] `load_model_set` 関数の実装（`model_set="primary_small"`）
  - [x] CLI オプション override 機構（優先順位: CLI > YAML > default）
  - [x] モデルメタデータ（`num_layers`, `hidden_dim`）の HF Config 動的取得機能とフォールバック
  - [x] 既存コードとの後方互換性担保（`get_registry()`, `get_family()`）
- [x] 4. **ModelAdapter の拡張 (`src/affective_empathy_eval/models/adapters.py`)**:
  - [x] Gemma 3 対応（`Gemma3Adapter`）
  - [x] OLMo 2 対応（`Olmo2Adapter`）
  - [x] `adapter` 名指定からの直接ディスパッチ対応
- [x] 5. **各 Stage スクリプトにおけるハードコード撤廃と `--model-set` 対応**:
  - [x] `behavioral/`, `v1/`, `v2/`, `v3/` の主要スクリプトにおける `MODEL_IDS` 等のハードコード排除
  - [x] `--model-set` / `--family` / `--base-model` / `--instruct-model` CLI 引数の共通受け入れ
- [x] 6. **統合 CLI エントリポイントの新設 (`src/affective_empathy_eval/run.py`)**:
  - [x] `python -m affective_empathy_eval.run --stage {behavioral,v1,v2,v3} --model-set primary_small`
- [x] 7. **テストと検証**:
  - [x] 新規 ModelRegistry / Adapter の単体テスト作成・実行
  - [x] 既存テストスイート（`.venv/bin/pytest -q`）の実行
  - [x] 統合 CLI による各 Stage の dry-run スモークテスト
  - [x] `walkthrough.md` の作成と保存
