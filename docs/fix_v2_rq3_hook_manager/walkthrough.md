# V2 RQ3 ActivationHookManager.get_captured 属性エラー解消 修正確認 (Walkthrough)

## 概要
`bash scripts/run_production_v2.sh cuda:0` 実行時、RQ1/RQ2（全 4 ファミリー）完了後の RQ3（因果マップ測定: `v2/primary/run_rq3_causal_map.py`）開始直後に発生した `AttributeError: 'ActivationHookManager' object has no attribute 'get_captured'` を解消しました。

---

## 主な変更内容

### 1. `ActivationHookManager` に `get_captured()` メソッドを追加
- **対象ファイル**: [`src/affective_empathy_eval/models/hooks.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/hooks.py)
- **変更点**:
  - キャプチャされた活性化テンソル辞書 `self.captured_activations` を取得するための標準メソッド `get_captured(self) -> Dict[str, torch.Tensor]` を実装。

### 2. `run_rq3_causal_map.py` の防御的呼び出し
- **対象ファイル**: [`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py)
- **変更点**:
  - `caps = hook_mgr.get_captured() if hasattr(hook_mgr, "get_captured") else getattr(hook_mgr, "captured_activations", {})` に更新し、メソッドおよび属性参照の双方に対応。

---

## 検証結果

### 単体テスト
[`tests/test_models_hooks.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_models_hooks.py) に `hook_mgr.get_captured()` が正しく動作し、`captured_activations` と等価なテンソル辞書を返却することを検証するアサーションを追加し、テストを実行しました。

```text
.venv/bin/pytest -v tests/test_models_hooks.py
============================== 1 passed in 2.13s ==============================
```

正常に PASS することを確認済みです。
