# タスク: 正式再実行前の V3 5点修正

## 必須

- [x] V3 既定データを AIPsy matched-neutral（`aipsy_4split_all.csv` の 192 pair）に合わせ、人工中立文と 5.0 fallback を禁止する
- [x] V3 RQ2 の生成段階 patch を `prompt_end` に丸めず、joint sequence 上の token で行う
- [x] V3 RQ2 / Confirmatory で Valence 方向と Arousal 方向の介入を分離する
- [x] Primary の β は符号付き係数と絶対値の両方を保存する
- [x] RQ1 Go/No-Go を production / 統合 CLI で実際に適用する（`--force-after-no-go` のみ継続）
- [x] 729 VAD と 81 VA の役割を decision_log と README で固定する

## 論文

- [x] 現行 README を正本にした構成メモを残す（旧 `iclr2027_conference2.tex` は旧稿として扱い、結果は捏造しない）
