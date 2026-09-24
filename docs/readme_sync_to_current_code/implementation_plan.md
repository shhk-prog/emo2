# 実装計画

実験コードは変更しない。README だけを、次の実装事実に合わせる。

- `configs/models.yaml` の `primary_small` / `scale_validation` / `scale_3b` / `scale_7b`
- `resolve_output_dirs` と `resolve_log_dir`
- `run.py` の stage 分岐。`--stage scale_validation` は confirmatory も model-set 上書きも転送しない
- `paper_summary` の 19 列と、各 Stage の `build_paper_summary.py` が実際に書く CSV
- V3 の `build_v3_summary` は `figure_data/` を作るが figure CSV は書かない
- V2 の図 CSV は `figure_v2_3_causal_relocation.csv` のみ
