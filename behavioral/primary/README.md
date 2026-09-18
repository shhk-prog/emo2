# Behavioral Primary

Behavioral の正式実行面。設計・指標・解釈の本文は親の [`behavioral/README.md`](../README.md) を正本とする。

| ファイル | 内容 |
|---|---|
| `run_behavioral_emobank.py` | EmoBank 3-Way。Writer / Reader / Self の独立セッション |
| `run_behavioral_aipsy.py` | AIPsy 4-Split。Sensitivity / dose-response / specificity / coupling |

候補空間は 729 VAD。集計は `behavioral/analysis/`。V3 は同じ AIPsy CSV から clinical–neutral pair だけを wide 化するが、Behavioral の 4-split 集計とは別指標である。
