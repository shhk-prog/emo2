# タスクリスト: V1 Phase C E3 NaN 集計エラーの修正とデータ復旧

- [x] エラーの根本原因調査 <!-- id: 0 -->
    - [x] `production_v1_20260920_193212.log` のエラー箇所の確認（`ValueError: Encountered all NA values`） <!-- id: 1 -->
    - [x] `v1/primary/run_phase_c.py` の集計コードの特定（`np.mean` による NaN 伝播） <!-- id: 2 -->
    - [x] 4時間12分の生計算データ `e3_causal_map_pair_level.csv` が完全・無傷であることを確認 <!-- id: 3 -->
- [x] コードの修正 <!-- id: 4 -->
    - [x] `v1/primary/run_phase_c.py` の E3 集計で `np.nanmean` および NaN フィルタリングを適用 <!-- id: 5 -->
    - [x] `v1/primary/run_phase_c.py` の E4 候補レイヤー選択で `dropna()` と空判定フォールバックを追加 <!-- id: 6 -->
    - [x] `v1/primary/phase_c/select_e4_sites.py` と `run_e6_specialization.py` にも `dropna()` ガードを追加 <!-- id: 7 -->
- [x] データの復旧 <!-- id: 8 -->
    - [x] `e3_causal_map_pair_level.csv` から正しい集計値で `e3_causal_map.csv` を再生成 <!-- id: 9 -->
    - [x] 再生成された `e3_causal_map.csv` の全26層に NaN がないことを確認 <!-- id: 10 -->
- [x] 検証とテスト <!-- id: 11 -->
    - [x] 既存テストスイートの実行（94 passed） <!-- id: 12 -->
    - [x] E4 の candidate selection が正常に動作することを確認 <!-- id: 13 -->
    - [x] `walkthrough.md` の作成と報告 <!-- id: 14 -->
