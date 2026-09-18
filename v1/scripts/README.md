# V1 Scripts Directory

本ディレクトリ内の過去の過渡期・探索用スクリプト群は、再現性と構成の明確化のため `v1/scripts/legacy/` 配下へ隔離・アーカイブされました。

現在本研究で公式に使用される V1 Primary 実装は、すべて **`v1/primary/`** 配下に集約・標準化されています。

## 正本スクリプト一覧
- `v1/primary/run_phase_a.py`: E1 Decodability & E2 Geometry (RSA / Cross-decoding)
- `v1/primary/run_phase_b.py`: Semantic vs. Lexical Controls Audit
- `v1/primary/run_phase_c.py`: E3 Prompt-End Causal Map & E4 Interchangeability
- `v1/primary/phase_c/run_e6_specialization.py`: E6 Double Dissociation & LMM
- `v1/primary/phase_c/summarize_phase_c.py`: Phase C クロスモデル要約生成

また、リポジトリルートの統合ランナー `python -m affective_empathy_eval.run --stage v1 --model-set primary_small` から全モデルへ一括実行可能です。
