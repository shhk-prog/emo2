# 修正内容の確認 (Walkthrough): V1 E6 アブレーションエラー & V3 RQ1 方向抽出戻り値エラーの修正

直近の実行ログで発生していた以下の2件のエラーを修正しました。

1. **V1 E6 アブレーションエラー** (`production_v1_20260920_023730.log`):
   ```text
     File ".../v1/primary/phase_c/run_e6_specialization.py", line 536, in main
       abl_vr_rs, _ = evaluate_expected_va_batch(...)
     File ".../src/affective_empathy_eval/intervention.py", line 107, in hook
       src = torch.tensor(self.source_tensor, device=hidden_states.device, dtype=hidden_states.dtype)
   TypeError: must be real number, not NoneType
   ```

2. **V3 RQ1 方向抽出戻り値型エラー** (`production_v3_20260920_174837.log`):
   ```text
     File "/mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py", line 346, in run_real_state_induction
       d_v = directions_primary["direction_v"]  # (D,) Primary Reader-grounded
   TypeError: tuple indices must be integers or slices, not str
   ```

---

## 1. V3 RQ1 エラー原因と修正内容

### エラー原因
- `v3/primary/run_rq1_state_induction.py`（および `run_rq3_path_mediation.py`）では、方向抽出関数の結果を `directions["direction_v"]`, `directions["direction_a"]` のように辞書形式で参照していました。
- しかし、`src/affective_empathy_eval/interventions.py` の `extract_conditional_directions` は単純なタプル `(d_v, d_a)` を返していたため、辞書インデックスアクセス時に `TypeError: tuple indices must be integers or slices, not str` が発生していました。

### 実施した修正
1. **[`src/affective_empathy_eval/interventions.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/interventions.py)**:
   - `ConditionalDirections` クラスを新設。`tuple` を継承しつつ、辞書スタイルアクセス（`["direction_v"]`, `["direction_a"]`, `["d_v"]`, `["d_a"]`）およびプロパティアクセス（`.direction_v`, `.direction_a`）をサポート。
   - これにより、タプルアンパック `d_v, d_a = extract_conditional_directions(...)` を行う既存のテストコード・スクリプトを一切壊すことなく、辞書アクセスを行う V3 スクリプト群とも完全な互換性を確保。
2. **[`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)**:
   - `unique_pairs = list(df["pair_id"].unique())` に修正し、PyArrow配列に対する `rng.shuffle` の UserWarning を解消。
3. **[`tests/test_interventions.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_interventions.py)**:
   - `ConditionalDirections` のタプルアンパック、インデックス参照、辞書キー参照、プロパティ参照を検証する単体テストを追加。

---

## 2. V1 E6 エラー原因と修正内容

### エラー原因
- E6（Targeted Ablation）では特定サイト（Reader-Site / Self-Site）の活動をゼロ置換（Zero Ablation）するため、`PyTorchActivationPatcher` を `source_tensor=None`, `intervention_type="zero"` として呼び出していました。
- しかし、フック内で無条件に `torch.tensor(self.source_tensor, ...)` を呼んでいたため `NoneType` 例外が発生していました。

### 実施した修正
1. **[`src/affective_empathy_eval/intervention.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/intervention.py)**:
   - `PyTorchActivationPatcher` において、`intervention_type in ("zero", "ablate_zero")` のゼロアブレーションをネイティブサポート（`patched = torch.zeros_like(original)`）。
2. **[`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)**:
   - ペア完了ごとに `e6_specialization_trials.csv` をディスクへ即時保存するインクリメンタル保存と `torch.cuda.empty_cache()` を追加。

---

## 3. 再実行コマンド

### V3 パイプライン再実行
```bash
bash scripts/run_production_v3.sh cuda:0
```

### V1 パイプライン再実行（Phase A/B/E3/E4 はスキップされ E6 から再開）
```bash
bash scripts/run_production_v1.sh cuda:0
```
