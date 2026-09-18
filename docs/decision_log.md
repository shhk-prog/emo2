# 決定ログ

実験仕様を変えたときは、変更理由・変更前後・影響範囲をここに残す。

| 日付 | 変更者 | 対象 | 変更前 | 変更後 | 理由 |
|---|---|---|---|---|---|
| 2026-09-18 | agent | V3 既定データ | `stimuli_vad_3way_test1k.csv`（EmoBank、pair なし） | `aipsy_4split_all.csv` の clinical–neutral 192 pair | matched-neutral / pair_id Group split とデータ定義を一致させる。人工中立文と 5.0 fallback を禁止するため。 |
| 2026-09-18 | agent | 候補空間の役割 | 文書上あいまい | Behavioral/V1 = 729 VAD、V2/V3 = 81 VA。両空間の VA 期待値は直接比較しない | 中心主張は VA だが、Behavioral/V1 は Dominance を含む 3 軸測定を維持する。コード上の Primary 候補は変更しない。 |
