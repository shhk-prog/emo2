# Task: 最終 Blocker 修正・本番即応化 (Final Blocker Fixes)

## 概要
ユーザーレビューで指摘された 5 大 BLOCKER および統計・再現性項目を順番に修正し、全 Stage の dry-run と pytest を完走させる。

## タスクリスト
- [ ] 1. `v1/primary/run_phase_c.py` に `import yaml` を追加
- [ ] 2. `behavioral/analysis/summarize_behavioral_aipsy.py` の glob 修正、0件時例外化 (`FileNotFoundError`)、モデル名抽出修正
- [ ] 3. `behavioral/primary/run_behavioral_aipsy.py` の dry-run を本番スキーマ完全再現に変更（`split`, `pair_id`, `triplet_id`, `emotion` 保持、deterministic な mock 生成）
- [ ] 4. Sequence-Likelihood を length-normalized (`normalize_length=True`) に統一
  - [ ] `src/affective_empathy_eval/likelihood.py` の default を `True` に変更
  - [ ] Behavioral, V1, V2, V3 の Primary 呼び出し確認・統一
  - [ ] configs YAML に `normalize_length: true` を明示
  - [ ] manifest に `sequence_likelihood_normalization: "token_mean"` を記録
- [ ] 5. キャッシュ無効化 (Cache Invalidation) の厳密化
  - [ ] `DEFAULT_CODE_VERSION` 更新
  - [ ] scoring 方式・normalization を manifest 照合に追加
- [ ] 6. `src/affective_empathy_eval/statistics.py` の `compute_d_z()` で `s_delta < 1e-9` 時に `np.nan` を返すよう修正
- [ ] 7. FDR family / direction map の明文化
- [ ] 8. 単体テスト実行 (`pytest -q`)
- [ ] 9. 全 Stage の dry-run 実行確認（Behavioral, V1, V2, V3）
- [ ] 10. `walkthrough.md` の作成と報告
