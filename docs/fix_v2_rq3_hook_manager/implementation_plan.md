# V2 RQ3 ActivationHookManager.get_captured 属性エラー解消 実装計画

## 概要
`bash scripts/run_production_v2.sh cuda:0` 実行時、RQ1/RQ2 は全 4 ファミリーで完了しましたが、後続の RQ3（因果マップ測定: `v2/primary/run_rq3_causal_map.py`）実行時に `AttributeError: 'ActivationHookManager' object has no attribute 'get_captured'` が発生しました。
本計画では、フック管理クラス `ActivationHookManager` に正規のインターフェースとして `get_captured()` メソッドを追加し、RQ3 スクリプト側からも安全に取得できるように整備します。

## 原因分析
- `v2/primary/run_rq3_causal_map.py` の 243 行目：
  ```python
  caps = hook_mgr.get_captured()
  ```
- 一方、`src/affective_empathy_eval/models/hooks.py` の `ActivationHookManager` では、キャプチャされた活性化テンソルが `self.captured_activations: Dict[str, torch.Tensor]` 属性として保持されていますが、アクセサメソッド `get_captured()` が定義されていませんでした。

## 修正方針

### 1. `ActivationHookManager` に `get_captured()` メソッドを追加
- `src/affective_empathy_eval/models/hooks.py`:
  ```python
  def get_captured(self) -> Dict[str, torch.Tensor]:
      """キャプチャされた活性化テンソルの辞書を取得"""
      return self.captured_activations
  ```

### 2. `v2/primary/run_rq3_causal_map.py` の防御的呼び出し
- `hook_mgr.get_captured()` を呼び出しつつ、属性直接参照にも安全に対応できるようにします。

## 変更対象ファイル
- **[MODIFY]** [src/affective_empathy_eval/models/hooks.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/hooks.py)
- **[MODIFY]** [v2/primary/run_rq3_causal_map.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py)

## 検証計画
- `tests/test_models_hooks.py` に `hook_mgr.get_captured()` の検証テストを追加して実行。
- `pytest tests/test_models_hooks.py` が成功することを確認。
