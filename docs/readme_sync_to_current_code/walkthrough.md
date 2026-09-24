# 確認

README を現行の実行面と論文集計面に合わせた。実験スクリプトは変更していない。

更新した文書:

- `README.md`
- `scripts/README.md`
- `behavioral/README.md`, `behavioral/primary/README.md`
- `v1/README.md`, `v1/primary/README.md`, `v1/scripts/README.md`
- `v2/README.md`, `v2/primary/README.md`, `v2/scripts/legacy/README.md`
- `v3/README.md`, `v3/primary/README.md`, `v3/scripts/README.md`, `v3/scripts/legacy/README.md`, `v3/docs/legacy/README.md`

コードと突き合わせた主な点:

- Primary 以外の V2 成果物は `results/ablation/{model_set}/`
- 統合 CLI は `resolve_log_dir` へログを書く
- 起動時インベントリは test1k、AIPsy、Phase B 統制の 3 ファイル
- 論文表は `build_all_paper_summaries.py` と各 Stage の `build_paper_summary.py`
- LaTeX は `scripts/generate_paper_results_tables.py` と `scripts/summarize_*.py`
