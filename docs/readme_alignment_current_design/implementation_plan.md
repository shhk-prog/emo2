# 実装計画: README を現行コードに揃える

正本はコードと `docs/v3_prerun_five_fixes/` である。旧論文 draft の数字は書かない。

## 共通で揃える用語

- Reader = 操作的な認識課題（平均的読者の VA を推定する）
- Self = 操作的な自己報告 / 反応性課題
- 認知的共感 / 情動的共感との一対一対応はしない
- 仮説図は一直線ではなく Shared representation → Reader / Self の分岐
- Behavioral / V1 = 729 VAD、V2 / V3 = 81 VA。数値を直接比較しない
- V1 の Base/Instruct は再現条件。差の解釈は V2
- V3 既定データは AIPsy clinical–neutral 192 pair
- V3 ゲートは完全一致の `GO` のみ継続

## 触るファイル

現行導線の README のみ。`.pytest_cache` は触らない。
