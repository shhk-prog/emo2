# 実装計画: V1 Phase C E3 NaN 集計エラーの修正とデータ復旧

## 背景と課題
`results/logs/production_v1_20260920_193212.log` において、`gemma_base` の E3（Causal Mapping）全26層の介入計算（所要時間：4時間12分、計4,992回のペア介入）は正常に完了し、ペアレベルの生データ `v1/results/derived/v1_phase_c_prompt_end/gemma_base/e3_causal_map_pair_level.csv` に保存された。
しかし、一部の刺激ペアで JSON パースや VAD 出力に欠損（NaN）が生じた際、`v1/primary/run_phase_c.py` 内のレイヤー集計処理（798〜884行目）で `np.mean` を直接呼んでいたため、レイヤー全体の集計値（`magnitude_self`, `magnitude_reader`, `discovery_mag_self`, `discovery_mag_reader` など）がすべて `NaN` に伝播してしまった。
その結果、後続の E4 候補レイヤー選択（923〜928行目）において：
```python
reader_peak_l = int(df_e3.loc[df_e3["discovery_mag_reader"].idxmax(), "layer"])
```
全行 NaN の Series に対する `idxmax()` が `ValueError: Encountered all NA values` を送出してパイプラインが停止した。

## 修正方針
1. **`v1/primary/run_phase_c.py` の修正**:
   - `mean_sv`, `mean_sa`, `mean_rv`, `mean_ra` の計算に `np.nanmean` を使用。
   - `disc_mag_s`, `disc_mag_r`, `conf_mag_s`, `conf_mag_r` の内包表記で NaN を除外して平均を計算。
   - E4 候補選択部で `df_e3["discovery_mag_reader"].dropna()` を使い、有効値が存在しない場合の中央層フォールバックを追加。
2. **`v1/primary/phase_c/select_e4_sites.py` および `run_e6_specialization.py` の防御強化**:
   - `idxmax()` を呼ぶ前に `.dropna()` およびフォールバック処理を追加。
3. **`gemma_base` の既存 E3 データの完全復旧**:
   - ディスク上に完全に保存されている `e3_causal_map_pair_level.csv`（4,992行、全26層）を読み込み、修正した集計関数で正しい `e3_causal_map.csv` を再生成。
   - 4時間12分の再計算を一切行うことなく、即座に E3 完了状態とする。
4. **動作検証**:
   - `e3_causal_map.csv` の全26層に有限値が入っていることを確認。
   - E4 候補レイヤー選択スクリプトが正しく候補層を出力できることを確認。
   - 既存ユニットテストを実行。
