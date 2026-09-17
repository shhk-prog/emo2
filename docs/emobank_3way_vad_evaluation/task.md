# タスク: EmoBank 3軸（Writer / Reader / Self）× 3次元（VAD）評価・集計システムの整備

- [x] EmoBank 3軸データ（Writer / Reader / Self-Report）× 3次元（VAD）評価の設計
- [x] 公式テスト分割（$N=1,006$）抽出スクリプト（`v1/scripts/prepare_3way_emobank.py`）の実装
- [x] 729候補バッチ対数尤度による3タスク推論スクリプト（`v1/scripts/run_3way_vad_evaluation.py`）の実装
- [x] 全8モデル一括実行バッチスクリプト（`v1/scripts/run_all_3way_vad.sh`）の実装
- [x] **集計スクリプト（`v1/scripts/summarize_3way_vad.py`）の改修** <!-- id: 0 -->
  - [x] モデルごとに「①Writer, ②Reader, ③Self」×「Valence, Arousal, Dominance」の9個の個別表を出力
  - [x] 各モデルの 3×3 サマリーマトリクス表の出力
  - [x] 全モデル横断の9大比較表（3タスク × 3次元）の出力
  - [x] タスク別（3表）および次元別（3表）の統合比較表の出力
  - [x] CSVおよびMarkdown形式での自動出力機能
- [x] **評価指標体系の洗練（Alignment / Calibration / Collapse / Coupling）の反映** <!-- id: 3 -->
  - [x] 各タスク（①, ②, ③）における連続尤度相関 ($r$) と Greedy相関 ($r_{greedy}$) の双方の計算
  - [x] ③ Self-Report における 3大評価軸（Alignment: $r$, Calibration: $\text{MAE}$, Collapse: $P(5,5,5)$）の体系化
  - [x] ④ 補助指標: Reader-Prediction $\leftrightarrow$ Self-Report 結合度 ($\text{Corr}(R, S)$) の統合
  - [x] 各モデルの9表および全体統合表への新指標の反映
- [x] 既存の実行結果（Qwen 2.5 1.5B Base）に対する集計テストと出力フォーマットの検証 <!-- id: 1 -->
- [x] ユーザーへの実行案内と出力レポートの提示 <!-- id: 2 -->
