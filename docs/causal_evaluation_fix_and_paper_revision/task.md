# タスク: 因果評価スクリプトのバグ修正と論文原稿の改訂

- [x] 1. 現状の調査と問題箇所の特定・整理 <!-- id: 0 -->
    - [x] `compute_likelihoods_for_candidates` の返り値とEMD計算バグの確認 <!-- id: 1 -->
    - [x] プローブターゲット（Peak-vs-Neutral intensity indicator）と原稿記述の乖離の確認 <!-- id: 2 -->
    - [x] 候補列尤度プロトコルの実装（単純和 vs 長さ正規化）の確認 <!-- id: 3 -->
- [x] 2. 実装計画書（implementation_plan.md）の作成とユーザー承認 <!-- id: 4 -->
- [x] 3. コードの修正 <!-- id: 5 -->
    - [x] `v2/src/likelihood.py` の修正（raw likelihood と length-normalized likelihood の両対応） <!-- id: 6 -->
    - [x] `v3/scripts/run_causal_localization_sweep.py` のEMDバグ修正およびコメント・出力表記の整合化 <!-- id: 7 -->
    - [x] `v3/scripts/run_within_model_positive_control.py` のEMDバグ修正 <!-- id: 8 -->
    - [x] 関連するその他スクリプトの点検 <!-- id: 9 -->
- [x] 4. 論文原稿（`v3/docs/paper.md`）の全面的改訂 <!-- id: 10 -->
    - [x] 確定結果として「0%の因果的不十分性」を書かない安全かつ堅牢な論構成へ改訂 <!-- id: 11 -->
    - [x] Peak-vs-Neutral intensity indicator R²=0.548 への修正 <!-- id: 12 -->
    - [x] 尤度プロトコル（raw sequence sum と length-normalized の区別）の追記 <!-- id: 13 -->
    - [x] Ridge/CKA/Retrieval/Center-collapse（Mahalanobis収縮）の精密な記述の維持・強化 <!-- id: 14 -->
    - [x] §6 Causal Intervention: Required Validation セクションの新設・整理 <!-- id: 15 -->
- [x] 5. 検証と成果物のまとめ（walkthrough.md） <!-- id: 16 -->
