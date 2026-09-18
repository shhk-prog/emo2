# V1 Primary

V1 の正式実行面。設計・指標・解釈の本文は親の [`v1/README.md`](../README.md) を正本とする。

比較軸は同一モデル内の Reader ↔ Self。Base / Instruct は再現条件であり、差の解釈は V2。

| ファイル | 内容 |
|---|---|
| `run_phase_a.py` | E1 decodability、E2 geometry |
| `prepare_v1_phase_b_controls.py` | Phase B 統制文の生成 |
| `run_phase_b.py` | rule-based controlled perturbation |
| `run_phase_c.py` | E3 / E4 |
| `phase_c/run_e6_specialization.py` | タスク選択性サイトと Confirmation。distinct site が無ければ No-Go |
| `phase_c/summarize_phase_c.py` | Phase C 横断要約 |

候補空間は 729 VAD。`v1/scripts/legacy/` は主解析に使わない。
