# v2およびv3のREADME修正タスク

- [x] 1. 現状のコードベース・設定ファイル・スクリプト群の精査と差異分析 <!-- id: 0 -->
  - [x] v2 のスクリプト（RQ1〜RQ4の体系、2x2 cross-decoding, causal map, recovery patching, 厳密マッチング等）
  - [x] v3 のスクリプト（RQ1〜RQ3、パイプライン、多層整列パッチング、Mood Congruency、因果局在スイープ、OTメトリクス等）
- [x] 2. 実装計画書（`implementation_plan.md`）の策定と提示 <!-- id: 1 -->
- [x] 3. `v2/README.md` の改定 <!-- id: 2 -->
  - [x] 最新のスクリプト群（`run_v2_2x2_cross_decoding.py`, `run_v2_2x2_causal_map.py`, `run_v2_recovery_patching.py`等）および4モデルファミリー体系の反映
  - [x] 実験パイプラインおよび再現手順の更新
- [x] 4. `v3/README.md` の新規作成 <!-- id: 3 -->
  - [x] v3 の学術的問い（内部状態の誘導、解離の検証、気分一致効果、多層多様体整列パッチング、生成時因果スイープ）
  - [x] `v3/src`（batch_likelihood, ot_utils, model_utils）および `v3/scripts` の詳細解説
  - [x] 実行手順（`run_v2_v3_full_pipeline.sh` や個別スクリプトの実行方法）
- [x] 5. 整合性検証および `walkthrough.md` の記録 <!-- id: 4 -->
