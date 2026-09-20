# タスク: V2 RQ3 実行時の ActivationHookManager.get_captured 属性エラー解消

## 状況
- `bash scripts/run_production_v2.sh cuda:0` 実行時、RQ1/RQ2（クロスデコーディング・幾何解析）は全 4 モデル（Qwen, Llama, Gemma, OLMo）で正常完了した。
- 続く RQ3（Causal Map: `v2/primary/run_rq3_causal_map.py`）の開始直後、Qwen のベースライン活性化収集で以下のエラーが発生し停止した：
  `AttributeError: 'ActivationHookManager' object has no attribute 'get_captured'`

## 目標
1. `src/affective_empathy_eval/models/hooks.py` の `ActivationHookManager` に `get_captured()` メソッドを追加し、キャプチャされた活性化テンソル辞書を安全に取得できるようにする。
2. `v2/primary/run_rq3_causal_map.py` 側でも防御的に `get_captured()` / `captured_activations` の両方に対応する。
3. 単体テスト（`tests/test_models_hooks.py`）に `get_captured()` のテストを追加し、回帰テストを実行・検証する。

## タスクリスト
- [x] 計画策定とタスクファイルの作成 (`docs/fix_v2_rq3_hook_manager/`) <!-- id: 0 -->
- [x] `ActivationHookManager` に `get_captured()` メソッドの実装 <!-- id: 1 -->
- [x] `v2/primary/run_rq3_causal_map.py` の防御的呼び出しの適用 <!-- id: 2 -->
- [x] 単体テストの実行・検証 <!-- id: 3 -->
- [x] 修正内容の確認 (Walkthrough) ドキュメントの作成 <!-- id: 4 -->
