# タスク: 4ファミリーの 3B/7B 帯スケーリング・アブレーション実験環境の構築

## 目的
Primary コホート（1〜1.5B帯: Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2）で得られた情動反応性および幾何・因果・回復特性の頑健性を検証するため、同様の4ファミリーについて 3B 帯および 7B 帯のモデル群をアブレーション（スケーリング検証）として追加し、評価可能な実験環境を構築する。

## 確定仕様
- **対象ファミリー**: Qwen, Llama, OLMo（※ユーザー指示により Gemma は除外）
- **scale_3b (3B帯)**:
  - Qwen: `Qwen/Qwen2.5-3B` / `Qwen/Qwen2.5-3B-Instruct`
  - Llama: `meta-llama/Llama-3.2-3B` / `meta-llama/Llama-3.2-3B-Instruct`
  - （※OLMo 2 は 3B モデルが存在しないため、3B帯は Qwen と Llama の 2 ファミリー）
- **scale_7b (7B帯)**:
  - Qwen: `Qwen/Qwen2.5-7B` / `Qwen/Qwen2.5-7B-Instruct`
  - Llama: `meta-llama/Llama-3.1-8B` / `meta-llama/Llama-3.1-8B-Instruct`
  - OLMo: `allenai/OLMo-2-1124-7B` / `allenai/OLMo-2-1124-7B-Instruct`

## タスク一覧
- [x] 1. **モデルレジストリと設定ファイルの拡張**
  - [x] `configs/models.yaml` に `scale_3b` および `scale_7b` コホートを追加
  - [x] `src/affective_empathy_eval/models/registry.py` の `KNOWN_MODEL_DIMS` に 3B/7B モデルの (num_layers, hidden_dim) を登録
- [x] 2. **出力ディレクトリ・ログディレクトリの自動分離実装**
  - [x] `src/affective_empathy_eval/io.py` に `resolve_output_dirs` および `resolve_log_dir` を追加
  - [x] Primary 成果物（`v2/results/`）とアブレーション成果物（`results/ablation/{model_set}/`）を自動分離
  - [x] Primary ログ（`results/logs/`）とアブレーションログ（`results/ablation/{model_set}/logs/`）を自動分離
  - [x] `src/affective_empathy_eval/run.py` でファイルロギングを自動化
  - [x] アブレーション本番 bash スクリプト `scripts/run_production_scale_ablation.sh` を新設
  - [x] `v2/primary/run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`, `run_confirmatory_analysis.py` に適用
- [x] 3. **単体テストと動作検証**
  - [x] `tests/test_model_registry_and_adapters.py` で `scale_3b`, `scale_7b` のロードテスト（14件合格）
  - [x] `tests/test_io_modular.py` で `resolve_output_dirs`, `resolve_log_dir` のテスト（合格、計18件）
  - [x] `scripts/run_production_scale_ablation.sh` の実機動作確認（ログが `results/ablation/scale_3b/logs/` に保存されることを確認）
- [x] 4. **実行用スクリプト・ドキュメントの整備**
  - [x] 実行方法（ワンコマンド実行 CLI）のドキュメント化
  - [x] `docs/ablation_scale_3b_7b/walkthrough.md` の作成
- [ ] 5. **GPU実行の確認と相談**
  - [ ] AGENTS.md 1.7条 に従い、実際のGPU実行前にユーザーへ相談・確認
