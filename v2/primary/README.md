# V2 Primary

V2 の正式実行面。設計・指標・解釈の本文は親の [`v2/README.md`](../README.md) を正本とする。

比較軸は Base ↔ Instruct である。Reader ↔ Self の解釈は V1、状態誘導の時空間は V3。

| ファイル | 内容 |
|---|---|
| `run_rq1_rq2_cross_decoding.py` | 幾何と Reader–Self sharing の Base/Instruct 差 |
| `run_rq3_causal_map.py` | $D(l)$ と実介入 $C(l)$、ピーク解離 |
| `run_rq4_recovery_patching.py` | Instruct 分布を Base へ戻す recovery |
| `run_confirmatory_analysis.py` | 横断 LMM / FDR（実行した場合） |

候補空間は 81 VA。データ既定は EmoBank 3-way test。V3 の AIPsy matched-neutral とは混ぜない。`v2/scripts/` と `v2/scripts/legacy/` は主解析に使わない。
