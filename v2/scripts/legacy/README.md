# V2 Legacy Scripts Archive

`v2/scripts/legacy/` は、開発初期の探索・旧 SAE・旧 2×2 Qwen 専用スクリプトを隔離した場所である。主解析には使わない。

現行の実験正本は [`v2/primary/`](../../primary/README.md) である。設計の本文は [`v2/README.md`](../../README.md)。親ディレクトリの `build_paper_summary.py` はこの legacy には入らない。derived を読んで論文 19 列を書く presentation である。

| 現行 RQ | 正本 (`v2/primary/`) | このディレクトリの旧稿 |
|---|---|---|
| RQ1 / RQ2 幾何と sharing | `run_rq1_rq2_cross_decoding.py` | `run_strict_cross_decoding.py`, `run_cross_decoding.py`, `run_v2_2x2_cross_decoding.py` |
| RQ3 因果マップ | `run_rq3_causal_map.py` | `run_patching_screening.py`, `run_circuit_patching.py`, `run_v2_2x2_causal_map.py` |
| RQ4 recovery | `run_rq4_recovery_patching.py` | `run_v2_recovery_patching.py`, `run_sae_patching.py` |
| 確証的統合 | `run_confirmatory_analysis.py` | `run_mixed_effects_coupling.py` |

旧稿の Gemma 2 / Primary Mistral / 729 と 81 の混在解釈は現行コホートに使わない。
