# Task: 本番再実行前の最終厳密化 (Final Pre-Rerun Rigorous Alignment)

## ステータス概要

### 【P0: 最優先・本番結果に直結】
- [ ] 1. V1 Phase C の layer indexing 1層ずれ修正 (`v1/primary/run_phase_c.py`) <!-- id: 0 -->
  - `extract_hidden_states()` で `get_block_hidden_state` または `outputs.hidden_states[block_idx + 1]` を使用し embedding (index 0) を除外
  - テスト `test_phase_c_block_index_mapping` を追加
- [ ] 2. V3 Confirmatory H2 を matched-neutral への注入に修正 (`v3/primary/run_confirmatory_replication.py`) <!-- id: 1 -->
  - `prompt_neu_self` を構築し、clean neutral からの変位 `shift = injected - clean_neu` を測定
  - H3 は endogenous relevance なので `prompt_aff_self` からの subspace removal を維持
- [ ] 3. V3 Confirmatory H2 の介入層を RQ1 (relative depth ≈ 0.5) と統一 (`v3/primary/run_confirmatory_replication.py`, `configs/v3_experiments.yaml`) <!-- id: 2 -->
  - `mid_layer` 一本槍を廃止し、`sufficiency_layer` (relative depth ≈ 0.5) を明示
  - `sufficiency_layer`, `temporal_map_layer`, `mediation_layer` に分離
- [ ] 4. V2 RQ3 に matched-plain causal map を追加 (`v2/primary/run_rq3_causal_map.py`) <!-- id: 3 -->
  - 条件に `Base plain`, `Instruct matched-plain`, `Instruct native-chat` の3条件を導入
- [x] 【P0-1】V1 Phase C の layer indexing 1層ずれ修正 (`v1/primary/run_phase_c.py`, `tests/test_refinement_suite.py`) <!-- id: 0 -->
- [x] 【P0-2】V3 Confirmatory H2 の Sufficiency を中立文への方向加算注入 (`clean_neu` からの変位) に修正 (`v3/primary/run_confirmatory_replication.py`) <!-- id: 1 -->
- [x] 【P0-3】V3 Confirmatory の layer 解決を介入目的に応じて分離・整合 (`v3/primary/run_confirmatory_replication.py`, `configs/v3_experiments.yaml`) <!-- id: 2 -->
- [x] 【P0-4】V2 RQ3 に matched-plain causal map を追加し、Confirmatory H3 を matched-plain Primary にする (`v2/primary/run_rq3_causal_map.py`, `v2/primary/run_confirmatory_analysis.py`) <!-- id: 3 -->
- [x] 【P0-5】V1 dry-run の出力先ディレクトリ完全分離 (`v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_e6_specialization.py`, `summarize_phase_c.py`, `src/affective_empathy_eval/run.py`) <!-- id: 4 -->

### 【P1: 統計・推定の頑健性と不整合解消】
- [x] 【P1-6】V2 Confirmatory H3 を matched-plain Primary にする (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 5 -->
- [x] 【P1-7】V2 Confirmatory H4 も matched-plain recovery AUC を Primary にする (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 6 -->
- [x] 【P1-8】V3 RQ2 の $C(l,t)$ を $\alpha=1.0$ (reference alpha) に固定 (`v3/primary/run_rq2_spatiotemporal_maps.py`) <!-- id: 7 -->
- [x] 【P1-9】V3 RQ1 の attenuation ratio で natural shift < 0.05 を除外 (`v3/primary/run_rq1_state_induction.py`) <!-- id: 8 -->
- [x] 【P1-10】V3 RQ3 で valid ratio sample が 0件のとき NaN にする (`v3/primary/run_rq3_path_mediation.py`) <!-- id: 9 -->
- [x] 【P1-11】V2 H1a geometry で matched 欠損時に RuntimeError (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 10 -->
- [x] 【P1-12】RSA は `rsa_similarity` と明記 (`v2/primary/run_confirmatory_analysis.py`, `v2/primary/run_rq1_rq2_cross_decoding.py`) <!-- id: 11 -->
- [x] 【P1-13】manifest / cache validation の厳格化 (`src/affective_empathy_eval/manifests.py`) <!-- id: 12 -->

### 【P2: 記述整合・テスト修正】
- [x] 【P2-14】`tests/run_all_v3_dryruns.py` を production runner 呼び出し形式に改修 (`tests/run_all_v3_dryruns.py`) <!-- id: 13 -->
- [x] 【P2-15】V3 $\beta$-map の emotion covariate 重複防止 (`src/affective_empathy_eval/data.py`) <!-- id: 14 -->

### 【検証・ドキュメント】
- [x] 【検証】全 pytest テストスイートの実行 (79 passed, 0 failed) <!-- id: 15 -->
- [x] 【検証】統合ランナー全ステージ dry-run 実行 & 本番 results クリーン性確認 <!-- id: 16 -->
- [x] 【報告】walkthrough.md の作成・更新と完了報告 <!-- id: 17 -->
- [ ] 18. ドキュメント保存 (`docs/final_pre_rerun_rigorous_alignment/`) <!-- id: 17 -->
