# Walkthrough: README を現行設計に揃える

現行コードと `docs/v3_prerun_five_fixes/` を正本にした。旧論文の数値は書いていない。

## 揃えた用語

- Reader / Self は操作的課題。共感概念との一対一対応はしない
- 仮説図は共有表現からの分岐
- Behavioral / V1 = 729 VAD、V2 / V3 = 81 VA
- V1 の Base/Instruct は再現条件
- V3 は AIPsy matched-neutral。ゲートは完全一致の `GO` のみ継続

## 更新した README

ルート、Behavioral、V1、V2、V3、各 `primary/`、`v1/scripts`、`v3/scripts`、`v2/scripts/legacy`、`v3/docs/legacy`、旧結果 archive。

V1 E6 は task-selectivity、distinct site が無ければ No-Go、heuristic fallback 禁止を本文に書いた。
`.pytest_cache` は触っていない。
