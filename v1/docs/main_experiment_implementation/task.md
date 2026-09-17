# タスクリスト: LLM情動反応性評価・本実験（Main Experiment）改訂版実装

## 概要
ユーザーフィードバックに基づく「機能的内部表現」検証の厳密化パイプライン実装。

## 実装ステップ

- [x] **1. コアモジュールとデータ管理の厳密化**
  - [x] `src/affective_empathy_eval/metrics.py` (Over-empathy削除、ADA等名称固定)
  - [x] `src/affective_empathy_eval/splits.py` [NEW] (stimulus_id grouped split, dev/test分離)
  - [x] `src/affective_empathy_eval/manifests.py` [NEW] (テンソル抽出のメタデータ管理)
  - [x] `src/affective_empathy_eval/controls.py` (統制・ラベル処理ロジックの修正)

- [x] **2. 内部表現抽出モジュールの厳密化 (`extraction.py`)**
  - [x] Phase B1 (hidden stateのみ) の抽出位置固定 (`stimulus_last_token`, `prompt_last_token`, `stimulus_mean_pool`, `first_generated_token_input`)
  - [x] float16保存とmanifest.pyへの連携

- [x] **3. プロービング & RSAモジュールの修正 (`probing.py`)**
  - [x] dev/test分離およびGroupKFoldへの対応
  - [x] 指標拡充 ($R^2$, MAE, RMSE, Pearson, Spearman) とPCA/正規化のtrain内fit
  - [x] 4種類のRSA計算（human_VA, reported_post, delta, expressed_response）

- [x] **4. 因果的介入モジュールの厳密化 (`intervention.py`)**
  - [x] Activation Patching (近接刺激対, Clean/Corrupted/Patched 3-run 必須化, Recovery計算)
  - [x] Ablation (Neutral replacementを主分析)
  - [x] Steering Vector (複数強度注入と単調性評価)

- [x] **5. 介入評価モジュールの実装 (`evaluation.py`)** [NEW]
  - [x] 直接的評価 (次トークンlogit, VA出力確率等)
  - [x] 生成的評価 (自由応答抽出など)

- [x] **6. スクリプト・設定の統合**
  - [x] `scripts/run_main_experiment.py` (Phase B1用更新)
  - [x] `scripts/run_probing.py` (dev/test分離対応)
  - [x] `scripts/run_causal_intervention.py` (evaluation.py対応)
  - [x] `scripts/validate_main_run.py` [NEW] (manifestログ整合性チェック)
  - [x] `scripts/summarize_main_run.py` [NEW]

- [ ] **7. テストと動作検証**
  - [x] ユニットテスト (`tests/`) の拡充・実行
  - [ ] 極小実モデル (`tiny-random-LlamaForCausalLM`等) による実機スモーク結合テスト
