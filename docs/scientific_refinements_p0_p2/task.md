# タスクリスト: 本実験前 科学的妥当性・P0/P1/P2不整合修正 (全22項目)

- [x] **Phase 1: V3 P0 バグ & VA 軸別 Gate**
  - [x] 1. `compute_orthonormal_subspace` の SVD rank-aware 化および呼び出し側 tuple アンパック修正 (`v3/primary/run_rq1_state_induction.py`, `v3/primary/run_rq3_path_mediation.py`, `src/affective_empathy_eval/interventions.py`)
  - [x] 2. `generate_control_directions` API 呼び出し修正（Valence / Arousal 別コントロール生成）
  - [x] 3. V3 RQ1 Gate の Valence / Arousal 完全軸別化（Specificity, Necessity, Topic TVD, Decision の VA 分離）

- [x] **Phase 2: Behavioral 統計検定 & FDR 修正**
  - [x] 4. RQ1 Primary 統計量（`aligned_p_value`）への FDR 適用、Writer / Dominance を Supplementary に分離
  - [x] 5. RQ3 存在しない `p_value` 参照の修正（`displacement_p_value` を Primary として FDR 適用）
  - [x] 6. RQ2 単調性検定の 2 段階 IUT（Intersection-Union Test）化と FDR 適用
  - [x] 7. expected direction 未定義時の raw 値 fallback 廃止（NaN 化）
  - [x] 8. RQ2 Secondary slope の別統計量 CI 単純平均バグ修正（独立 bootstrap CI 計算）

- [x] **Phase 3: V2 RQ3 Cross-Fitting & Confirmatory 安全化**
  - [x] 9. V2 RQ3 affect direction 推定の 5-fold CV cross-fitting 化
  - [x] 10. V2 Confirmatory の実機実行時 mock fallback 完全排除（`FileNotFoundError` / `RuntimeError` 送出）
  - [x] 11. V2 Confirmatory の H1〜H4 全仮説集約実装（post-training-associated reorganization 表現の厳守）
  - [x] 12. V2 Confirmatory を production runner (`src/affective_empathy_eval/run.py`) に接続

- [x] **Phase 4: Dry-run / Cache 分離 & Manifest 照合接続**
  - [x] 13. `results/dry_run/` と `results/raw/` の完全分離、`dry_run: true` メタデータ保存
  - [x] 14. V2 / V3 全スクリプトにおける `is_manifest_matching` の実接続と filepath 引数への統一

- [x] **Phase 5: V3 Confirmatory 統計・評価の精密化**
  - [x] 15. H1 / H3 の Valence / Arousal 両軸化
  - [x] 16. Attenuation ratio の `[0, 1]` clip 撤廃（unclipped raw ratio + bootstrap CI）
  - [x] 17. 閾値バイナリ判定から効果量・bootstrap CI 中心報告への移行
  - [x] 18. 先頭5件固定抽出から事前固定シード・pair_id ベース抽出への修正
  - [x] 19. H4 Temporal Emergence の論文主張（decodability ≠ uniform causal leverage）との整合

- [x] **Phase 6: pytest 設定 & ドキュメント整合**
  - [x] 20. `pyproject.toml` の `pythonpath = ["src", "."]` 追加
  - [x] 21. root `README.md` の重複見出し削除
  - [x] 22. V2 README/文書における「モデル内因果介入」と「post-training observational 関連付け」の明瞭な分離

- [x] **Phase 7: テスト実行・検証・Walkthrough 作成**
  - [x] pytest 全体実行（通常テスト 75 passed + production entrypoints 3 passed = 計 78 passed）
  - [x] `docs/scientific_refinements_p0_p2/walkthrough.md` の作成
