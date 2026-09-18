# タスクリスト: Scale Validation (Mistral 7B) の Ablation 独立分離

## 目的
Mistral 7B による外部スケール検証（Scale Validation）を、Primary 1〜1.5B パイプラインから完全に切り離し、独立した Ablation 実験枠（`scale_validation`）として設定・スクリプト・実行コマンドを分離・整備する。

---

## タスク一覧

- [x] 1. **ドキュメント準備**:
  - [x] `docs/scale_validation_ablation_refactor/task.md` 作成
  - [x] `docs/scale_validation_ablation_refactor/implementation_plan.md` 作成・承認
- [x] 2. **設定の独立化 (`configs/`)**:
  - [x] `configs/scale_validation.yaml` を新設（Mistral 7B の Base/Instruct および Ablation 用評価設定）
  - [x] `configs/models.yaml` を純粋な Primary コホート（Qwen, Llama, Gemma, OLMo）に特化
- [x] 3. **独立実行スクリプトの新設**:
  - [x] `scripts/run_scale_validation.py` の新設（Mistral 7B による幾何・因果・分布回復の単独実行）
  - [x] 出力先を `results/ablation/scale_validation/` または各ステージの `results/ablation/` に分離
- [x] 4. **統合 CLI (`affective_empathy_eval.run`) の拡張**:
  - [x] `--stage scale_validation` または `--ablation scale_validation` でワンコマンド実行可能に
- [x] 5. **テストと検証**:
  - [x] `pytest -q` の確認
  - [x] `python scripts/run_scale_validation.py --dry-run` の動作確認
  - [x] `python -m affective_empathy_eval.run --stage scale_validation --dry-run` の動作確認
  - [x] `walkthrough.md` の作成と保存
