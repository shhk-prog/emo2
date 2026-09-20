# 修正内容の確認 (Walkthrough): V1 Phase C E3 NaN 集計エラー修正とデータ復旧

## 1. 概要
`results/logs/production_v1_20260920_193212.log` において、`gemma_base` の E3 Causal Mapping（全26層×192ペア＝4,992回の介入計算、所要時間：4時間12分）が完了した直後、後続の E4（Interchangeability）へ移行する際に以下のエラーが発生し停止しました：
```text
ValueError: Encountered all NA values
  File "v1/primary/run_phase_c.py", line 924, in <module>
    reader_peak_l = int(df_e3.loc[df_e3["discovery_mag_reader"].idxmax(), "layer"])
```

## 2. 根本原因の特定
- Gemma-Base では、一部の刺激ペアで JSON パースや VAD 出力に欠損が生じていました（`NaN`）。
- `v1/primary/run_phase_c.py` の 798〜801行目および 827〜884行目のレイヤー集計処理において、リスト内に NaN が存在する場合の考慮がなく、通常の `np.mean` を呼び出していました。
- NumPy の仕様により、1つでも NaN を含む配列の `np.mean` は `nan` となるため、`magnitude_self`, `magnitude_reader`, `discovery_mag_self`, `discovery_mag_reader` などの全層の集計値がすべて `NaN` となって出力されていました。
- その結果、全行が NaN の `discovery_mag_reader` に対して `idxmax()` を呼んだため `ValueError: Encountered all NA values` が発生しました。

## 3. データの保全確認
- ペアレベルの生測定結果である `v1/results/derived/v1_phase_c_prompt_end/gemma_base/e3_causal_map_pair_level.csv`（4,992行、全26層分）はディスク上に**完全に無傷で保存**されていました。
- したがって、4時間12分の GPU 計算を最初からやり直す必要は全くありませんでした。

## 4. 実施した修正

### (1) `v1/primary/run_phase_c.py` の修正
- ペア単位のシフト平均（`mean_sv`, `mean_sa`, `mean_rv`, `mean_ra`）計算時に NaN を安全にフィルタリングして算出するよう修正。
- `discovery_mag_s`, `discovery_mag_r`, `confirmation_mag_s`, `confirmation_mag_r` の計算でも NaN ペアを除外して平均を算出するよう修正。
- E4 候補レイヤー選択部（`reader_peak_l`, `self_peak_l`）で `.dropna()` を適用し、有効値がない場合の中央付近レイヤーへのフォールバックを実装。

### (2) 関連スクリプトの防御強化
- `v1/primary/phase_c/select_e4_sites.py`: `idxmax()` 呼び出し前に `.dropna()` と空判定フォールバックを追加。
- `v1/primary/phase_c/run_e6_specialization.py`: タスク選択性コントラスト（`selectivity_r`, `selectivity_s`）の算出時に `.dropna()` を適用し、すべて NaN の場合の安全な No-Go 判定を実装。

### (3) `gemma_base` の E3 レイヤー集計データの完全復旧
- スタンドアロンの復旧スクリプト `scripts/repair_e3_causal_map.py` を作成・実行。
- 保存済み生データ（4,992行）から正しい集計値を算出し、`e3_causal_map.csv` を再生成（全26層すべてで非欠損値を確認）。

## 5. 検証結果
1. **復旧データの検証**:
   - `e3_causal_map.csv` の全26層において欠損値がゼロ（0 NaN）であることを確認。
2. **候補レイヤー選択の動作確認**:
   - `select_e4_sites.py` を修復済み `e3_causal_map.csv` で実行し、`Selected E4 candidate layers: [0, 1, 2, 22]` が正常に出力・保存されることを確認。
3. **ドライランテスト**:
   - `run_phase_c.py --family gemma --model-prefix gemma_base --mode all --dry-run` が exit code 0 で正常終了することを確認。
4. **全テストスイートの実行**:
   - `pytest -q tests/`: **94 passed, 1 deselected** で全テストが成功。
