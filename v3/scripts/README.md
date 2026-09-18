# V3 Scripts Directory

本ディレクトリ内の過去の過渡期・探索用スクリプト群は、再現性と構成の明確化のため `v3/scripts/legacy/` 配下へ隔離・アーカイブされました。

現在本研究で公式に使用される V3 Primary 実装は、すべて **`v3/primary/`** 配下に集約・標準化されています。

## 正本スクリプト一覧
- `v3/primary/run_rq1_state_induction.py`: RQ1 State Induction & Go/No-Go Gate
- `v3/primary/run_rq2_spatiotemporal_maps.py`: RQ2 4-Map Spatiotemporal Geometry & Causality
- `v3/primary/run_rq3_path_mediation.py`: RQ3 Discovery-Confirmation Path Mediation
- `v3/primary/run_confirmatory_replication.py`: Step 7 Confirmatory Replication across Families

また、リポジトリルートの統合ランナー `python -m affective_empathy_eval.run --stage v3 --model-set primary_small` から全モデルへ一括実行可能です。
