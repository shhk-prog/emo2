# タスクリスト: 本実験前 最終科学的妥当性・整合性修正 (全25項目)

- [ ] **Phase 1: V2 Confirmatory & 幾何スキーマ修正 (項目 1〜5, 19〜21, 25)**
  - [ ] 1. 【P0】V2 Confirmatory H1/H2 スキーマ修正 (`v2/primary/run_confirmatory_analysis.py`)
  - [ ] 2. 【P0】V2 H1 判定の緩和（後方シフト固定から雙方向再編・系統的変化へ、`reorganization_supported = ci_low > 0 or ci_high < 0`）
  - [ ] 3. 【P0】V2 H2 `delta_sharing_matched` 直接利用（Primary: Base vs Matched-plain, Secondary: Base vs Native-chat）
  - [ ] 4. 【P0】`v2_geometry` 返り値に `relative_depths` と `num_layers` を保存 (`v2/primary/run_rq1_rq2_cross_decoding.py`)
  - [ ] 5. 【P0】V2 RQ3 cache hit 時の pair-level CSV 消失防止（family別CSV保存とcache照合、`v2/primary/run_rq3_causal_map.py`)
  - [ ] 19. 【P1】V2 Confirmatory H3 名称・解釈修正 (`H3_causal_profile_reorganization_lmm`)
  - [ ] 20. 【P1】V2 H3 LMM への family 固定効果導入 (`c ~ C(family) + C(alignment) * C(task) * relative_depth`)
  - [ ] 21. 【P1】V2 H3 Primary FDR 対象項の事前固定 (`PRIMARY_TERMS`)
  - [ ] 25. 【P2】V2 コードコメント内の表現修正（「純粋post-training効果」の撤廃）

- [ ] **Phase 2: V1 E4 感情方向アラインメント & 伝達比率 (項目 6〜7)**
  - [ ] 6. 【P0】V1 E4 感情正負相殺の解消（`EXPECTED_DIRECTION` に基づく target direction aligned effect の Primary 化、`v1/primary/run_phase_c.py`）
  - [ ] 7. 【P1】V1 E4 Transfer Ratio の aligned effect 化と微小分母ガード (`abs(mean_ss) < 0.05` で NaN）

- [ ] **Phase 3: V3 RQ3 Mediator 選定 & 減衰比率安定化 (項目 8〜11, 13)**
  - [ ] 8. 【P0】V3 RQ3 Mediator 選定の先頭8件固定排除（8感情 stratified sampling + seed固定）
  - [ ] 9. 【P0】V3 RQ3 Mediator 選定の Valence/Arousal 両軸化（$C_V(l), C_A(l), C_{\text{joint}}(l)$ による層選定）
  - [ ] 10. 【P1】V3 RQ3 attenuation ratio の微小変位ガード（`MIN_NATURAL_SHIFT = 0.05`、Primary は absolute mediated attenuation）
  - [ ] 11. 【P1】V3 Confirmatory 減衰比率安定化（`natural_shift > 0.05` ガード）
  - [ ] 13. 【P1】V3 RQ3 `mu_neu` fallback の完全削除（matched-neutral 必須化）

- [ ] **Phase 4: V3 RQ2 トークン位置不変性 & バグ修正 (項目 12, 14, 15, 22)**
  - [ ] 12. 【P0】V3 Confirmatory matched-neutral fallback 削除（存在しない場合は例外送出）
  - [ ] 14. 【P1】V3 RQ2/Confirmatory 共通の `validate_stage_index_invariance` 接続 (`src/affective_empathy_eval/prompts.py`)
  - [ ] 15. 【P1】V3 RQ2 重複代入バグの削除 (`C_A[l, s_idx]`)
  - [ ] 22. 【P1】V3 Confirmatory cross-family summary への H4 定量コントラスト集約

- [ ] **Phase 5: キャッシュ・Manifest・Dry-run 出力分離 (項目 16〜18)**
  - [ ] 16. 【P0】dry-run と real 結果の保存先完全分離（`results/dry_run/` vs `results/raw/`）
  - [ ] 17. 【P1】manifest validation 厳格化（config hash, dataset hash, code version 照合）
  - [ ] 18. 【P1】cache hit 時の manifest 上書き洗浄防止

- [ ] **Phase 6: ドキュメント & README 統計記述修正 (項目 23〜24)**
  - [ ] 23. 【P2】Behavioral README の統計検定記述修正（Jonckheere-Terpstra 削除、IUT 表記へ）
  - [ ] 24. 【P2】Behavioral Sensitivity paired t 表記の精密化（Direction-aligned paired difference）

- [ ] **Phase 7: テスト実行・検証・Walkthrough 作成**
  - [ ] `pytest` 全体実行・回帰確認
  - [ ] `docs/scientific_refinements_final_pre_run/walkthrough.md` 作成
