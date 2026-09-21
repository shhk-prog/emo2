# 決定ログ

実験仕様を変えたときは、変更理由・変更前後・影響範囲をここに残す。

| 日付 | 変更者 | 対象 | 変更前 | 変更後 | 理由 |
|---|---|---|---|---|---|
| 2026-09-18 | agent | V3 既定データ | `stimuli_vad_3way_test1k.csv`（EmoBank、pair なし） | `aipsy_4split_all.csv` の clinical–neutral 192 pair | matched-neutral / pair_id Group split とデータ定義を一致させる。人工中立文と 5.0 fallback を禁止するため。 |
| 2026-09-18 | agent | 候補空間の役割 | 文書上あいまい | Behavioral/V1 = 729 VAD、V2/V3 = 81 VA。両空間の VA 期待値は直接比較しない | 中心主張は VA だが、Behavioral/V1 は Dominance を含む 3 軸測定を維持する。コード上の Primary 候補は変更しない。 |
| 2026-09-21 | agent | token-level continuation boundary 定義 | 未定義（Qwen2.5 等の BPE トークナイザーで `" {"` マージによりクラッシュ） | `canonicalize_prompt_candidate_boundary` を `likelihood.py` に導入。prompt 末尾 trailing ASCII space を candidate 先頭へ移動した上で `require_strict_prefix=True` による strict prefix 検証を維持。連結文字列（モデルへの完全入力列）は不変。token-level continuation boundary を明示的に再定義するため、`" {"` 等の trailing whitespace を含む先頭トークンが candidate 側の length normalization 対象に含まれる。旧実装（クラッシュ）との数値的一致は保証しない。Behavioral/V1/V2/V3 全 Stage・全モデルで統一適用。本変更後は全 Stage fresh run が必要。 | Qwen2.5-1.5B BPEマージにより RQ3 パイプラインが `ValueError` で停止。`require_strict_prefix=False`（LCP フォールバック）は length-normalized metric の分母を曖昧にするため採用しない。`build_prompt()` を変更すると過去実験との比較条件が変わるため採用しない。 |
| 2026-09-21 | hiromi | V3 RQ1 Gate NO_GO 後の強制続行 | V3 RQ1 Gate = NO_GO（dose_response・specificity・endogenous_relevance・necessity すべて不合格、topic_control のみ合格）でパイプライン停止 | `--force-after-no-go` フラグで RQ2/RQ3/Confirmatory を続行実行（`bash scripts/run_production_v3.sh cuda:0 --force --force-after-no-go`）。**主分析の結論は NO_GO のまま変更しない。** RQ2/RQ3 以降の結果は「Gate 不合格後の探索的分析」として位置づけ、論文では主分析と明確に分離して記述する。 | RQ2/RQ3 の探索的結果を将来研究・付録材料として確認するため。Gate 判定自体の覆しではない。AGENTS.md §7.4「事後分析を事前仮説であるかのように記述すること」は禁止。 |
