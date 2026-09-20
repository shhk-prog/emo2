# タスクリスト: 最終本番前26項目監査

## 監査項目リスト
- [x] 1. 構文・テスト検証 (`compileall`, `pytest -q`, 特定7テスト)
- [x] 2. V1 Phase B の `--force` 引数が1個だけであることの確認
- [x] 3. Behavioral/V1 へ model revision が runner から伝播していることの確認
- [x] 4. `configs/models.yaml` の Primary revision が commit SHA 固定であることの確認
- [x] 5. 全 Primary loader が `revision=` を使用していることの確認
- [x] 6. V1 Phase B の manifest config を hash 後に変更していないこと (`num_pairs` 等)
- [x] 7. Candidate-space sensitivity の Clinical 選択が厳密であること (`isin(["peak", "clinical"])`)
- [x] 8. Candidate-space sensitivity が model revision を使用していること
- [x] 9. Candidate-space sensitivity で結論が hard-code されていないこと
- [x] 10. V3 の candidate space 表記が `VA_81` であること
- [x] 11. V1 Phase C/E6 が `VAD_729` であること
- [x] 12. V3 RQ2/RQ3 cache で revision を照合していること
- [x] 13. V3 production で pair-aware split から fallback しないこと (`raise ValueError`)
- [x] 14. negative R² を 0 へ clip していないこと
- [x] 15. NaN/Inf hidden state を 0 へ書き換えていないこと
- [x] 16. V2 RQ4 の ΔEMD が sample-wise であること
- [x] 17. V2 RQ4 が AUC を Primary、best-layer/max を Secondary にしていること
- [x] 18. V1 E4 の confirmatory site が Discovery だけで選ばれていること
- [x] 19. V3 frozen site が V/A 別であること (`frozen_confirmatory_sites.json`)
- [x] 20. V3 Confirmatory で site を再探索していないこと
- [x] 21. Behavioral/V1/V2/V3 で prompt hash が SHA-256 であること
- [x] 22. candidate hash も実候補全件（729 / 81）から作られていること
- [x] 23. dtype が意図した条件 (`bfloat16`) に揃っていること
- [x] 24. 旧結果と新結果が混ざらない設計（アーカイブ体制）になっていること
- [x] 25. 秘密ファイル (`.env` 等) が Git / 配布物から除外されていること
- [x] 26. 統合 dry-run (`affective_empathy_eval.run`) が完全完走すること
- [x] 総合判定（GO / NO-GO）および結果報告
