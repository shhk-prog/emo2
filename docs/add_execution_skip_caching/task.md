# タスク: 全ステージにおける実行結果自動スキップ機能の実装

## 目的
現在 V2 および V3 の主要コンポーネント（RQ2, RQ3, Confirmatory）に実装されていた「成果物およびマニフェストが存在し検証通過した場合の自動スキップ機能」を、**Behavioral ステージ**、**V1 ステージ**、および **V3 RQ1** にも統一的に導入した。
これにより、長時間かかる重みロードやモデル推論の不要な重複実行を回避し、中断からの安全な再実行（冪等性）が可能となった。同時に、意図的に再計算を行いたい場合のための `--force` オプションを全ステージで統一実装した。

---

## 完了した作業

1. **Behavioral ステージ**:
   - [x] `behavioral/primary/run_behavioral_emobank.py`:
     - 最終出力 CSV（`{tag}_3way_vad.csv`）が存在し有効な場合、モデルロード前にスキップする判定を追加。
     - `--force` 引数を追加（指定時はキャッシュをバイパスして再計算）。
   - [x] `behavioral/primary/run_behavioral_aipsy.py`:
     - 最終出力 CSV（`{tag}_aipsy_4split.csv`）が存在し有効な場合、モデルロード前にスキップ。
     - `--force` 引数を追加。

2. **V1 ステージ**:
   - [x] `v1/primary/run_phase_a.py`:
     - 成果物（`e1_emobank_decodability.csv`, `e2_emobank_geometry.csv`, `manifest.json`）が存在し有効な場合、モデルロード前にスキップ。
     - `--force` 引数を追加。
   - [x] `v1/primary/run_phase_b.py`:
     - 成果物（`phase_b_semantic_controls.csv`, `manifest.json`）が存在し有効な場合、モデルロード前にスキップ。
     - `--force` 引数を追加。
   - [x] `v1/primary/run_phase_c.py`:
     - 成果物（`e3_causal_map.csv`, `e4_interchangeability_results.csv`, `manifest.json`）が存在し有効な場合、モデルロード前にスキップ。
     - `--force` 引数を追加。
   - [x] `v1/primary/phase_c/run_e6_specialization.py`:
     - 成果物（`e6_lmm_results.json`, `manifest_e6.json`）が存在し有効な場合、モデルロード前にスキップ。
     - `--force` 引数を追加。

3. **V2 ステージ**:
   - [x] `v2/primary/run_rq1_rq2_cross_decoding.py`: `--force` 引数を追加し、指定時は既存キャッシュをバイパスして再計算。
   - [x] `v2/primary/run_rq3_causal_map.py`: `--force` 引数を追加。
   - [x] `v2/primary/run_rq4_recovery_patching.py`: `--force` 引数を追加。

4. **V3 ステージ**:
   - [x] `v3/primary/run_rq1_state_induction.py`: `v3_rq1_results.json`, `v3_gate_decision.json`, `manifest_rq1_{fam_key}.json` が存在し有効な場合、モデルロード前にスキップ。`--force` 引数を追加。
   - [x] `v3/primary/run_rq2_spatiotemporal_maps.py`: `--force` 引数を追加。
   - [x] `v3/primary/run_rq3_path_mediation.py`: `--force` 引数を追加。
   - [x] `v3/primary/run_confirmatory_replication.py`: `--force` 引数を追加。

5. **統合ランナー (`affective_empathy_eval/run.py`)**:
   - [x] `--force` 引数を追加。
   - [x] 各ステージ（`run_behavioral`, `run_v1`, `run_v2`, `run_v3`）のサブコマンドへ `--force` を透過的に伝達。
   - [x] `batch_size` 属性の安全な参照（`getattr(args, "batch_size", None)`）へ修正。

6. **V1 Phase C (E3 / E4) 進捗可視化・インクリメンタル保存・スキップ強化**:
   - [x] `v1/primary/run_phase_c.py`:
     - E3 完了時のスキップ表示を強調枠線＋即時フラッシュ（`flush=True`）に変更。
     - E3 層単位のインクリメンタル保存・再開機能の追加。
     - E4 に条件単位（`E4 Conditions`）およびペア単位（`E4 L{l} a={alpha}`）の `tqdm` プログレスバーを追加。
     - E4 に条件単位のインクリメンタルディスク保存と `empty_cache()` を追加。
     - E4 全条件完了時の強調スキップ表示を追加。

7. **テスト & 検証**:
   - [x] `py_compile`: 対象全スクリプトのエラーゼロ確認。
   - [x] `tests/test_execution_skip_caching.py`: 4件のテストが全件 PASS。
   - [x] `pytest -q`: 全89テストが PASS（回帰ゼロ）。
