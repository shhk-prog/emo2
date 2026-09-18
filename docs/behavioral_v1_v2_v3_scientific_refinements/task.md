# タスクリスト: Behavioral / V1 / V2 / V3 科学的・実験的リファインメント

## 1. V3 Direction Intervention の重大修正
- [x] `src/affective_empathy_eval/models/hooks.py` に `register_direction_intervention_hook` または統一 `apply_direction_intervention` API（additive injection: $h' = h + \alpha \cdot \sigma_h \cdot \hat{d}$, replace: $h' = \alpha \cdot \sigma_h \cdot \hat{d}$）を実装・洗練
- [x] `src/affective_empathy_eval/likelihood.py` の `generation_patch` に `mode` ("inject" / "replace") を明示導入し additive injection をサポート
- [x] `v3/primary/run_rq1_state_induction.py` の全置換（`register_patch_hook`）を排除し、完全な additive injection（$h + \alpha \cdot \sigma_h \cdot \hat{d}$）に統一
- [x] `v3/primary/run_rq2_spatiotemporal_maps.py` の generation patch 介入を additive injection に統一
- [x] `v3/primary/run_confirmatory_replication.py` の介入を同一オペレータ（$\alpha \cdot \sigma_h$ のスケール定義統一）に揃える

## 2. V3 Intervention Sanity Tests
- [x] `tests/test_v3_interventions_sanity.py` を作成し、以下6項目を検証:
  1. $\alpha = 0$ で baseline 出力と一致
  2. inject で $h' - h = \alpha \cdot \sigma_h \cdot \hat{d}$ と完全一致
  3. replace で指定ベクトルへ置換
  4. direction norm を変えても unit normalize 後の介入量は不変
  5. random / orthogonal / affective direction で scale 定義が同一
  6. Confirmatory と Discovery で同一の $\alpha$ が同一の activation norm change を生む

## 3. Behavioral Reader-Self Coupling の修正
- [x] `behavioral/analysis/summarize_behavioral_aipsy.py` において、Primary metric を raw 相関から matched pair 刺激変化に対するカップリング $\text{corr}(\Delta_{reader}, \Delta_{self})$ へ刷新
  - $\Delta_{reader} = reader_{affective} - reader_{neutral}$
  - $\Delta_{self} = self_{affective} - self_{neutral}$
- [x] Valence と Arousal を分離計算
- [x] raw correlation は secondary / supplementary として保持
- [x] pair_id 単位の bootstrap で 95% CI を算出
- [x] 出力 CSV カラムを仕様に準拠 (`model, alignment, dimension, n_pairs, correlation_type, r, ci_low, ci_high, p_value`)

## 4. Behavioral Dose-Response 解析の修正
- [x] 3条件縦積み独立観測を Primary から外し、同一 triplet/pair 内の反復測定構造を保持する設計へ変更（triplet 単位の linear slope / ordered within-triplet contrast）
- [x] 統計的独立単位を pair_id / triplet_id に固定
- [x] bootstrap を triplet 単位で実行
- [x] BH-FDR の補正対象 family をコードとドキュメントで明確化

## 5. Behavioral Bootstrap CI
- [x] paired mean difference, Cohen's $d_z$, delta Reader-Self coupling の 95% CI を計算・保存
- [x] 既存の `compute_bootstrap_ci` を再利用・拡張

## 6. Layer Indexing の全面統一
- [x] 共通 utility `get_block_hidden_state(hidden_states, layer_idx)` を作成（$0 \dots L-1$、`hidden_states[0]` は embedding、$l$ 番目ブロック出力は `hidden_states[l + 1]`）
- [x] relative depth の定義を `layer_idx / (num_hidden_layers - 1)` に完全統一
- [x] V1 Phase A/B/C, V2, V3 の layer 抽出コードをすべて共通 utility 経由に統一

## 7. V1 Phase B の Generalization 設計
- [x] "held-out semantic generalization" の呼称を "semantic transformation sensitivity" 等の実態に即した表現へ変更
- [x] pair-aware evaluation（train: training pair_ids の original, test: held-out pair_ids の paraphrase/reversal）を追加実装し、同一 pair 由来データのリーク防止 assert を導入

## 8. V2 Aligned Activation Patch Control
- [x] `v2/primary/run_rq4_recovery_patching.py` に以下の2条件を明示導入:
  - Condition A: Direct Base -> Instruct activation patch
  - Condition B: Aligned Base -> Instruct activation patch (train 側のみから学習した Procrustes alignment を適用)

## 9. V2 Bootstrap Config Bug 修正
- [x] `configs/v2_experiments.yaml` の `bootstrap.n_boot` を正しく読み込むよう `v2/primary/` コードを修正（`statistics.n_boot` 参照バグ解消）

## 10. V3 Confirmatory の Data Reuse 防止
- [x] `v3/primary/run_confirmatory_replication.py` において、direction 推定と intervention 評価を同一サンプルで行わないよう pair_id 単位の cross-fitting / holdout を実装
- [x] 最終結果は test fold のみ aggregate

## 11. Result Cache の安全化
- [x] `src/affective_empathy_eval/manifests.py` に `intervention_version`、`git_commit`、`config_hash` 等の検証ロジックを実装
- [x] 旧 replacement キャッシュと新 additive injection キャッシュの混在を防止

## 12. Output Directory の整理
- [x] `results/raw/`, `results/derived/`, `results/figures/`, `results/tables/`, `results/manifests/` の構造に Behavioral を含めて統一

## 13. Behavioral Unified Runner
- [x] `python -m affective_empathy_eval.run --stage behavioral` のみで raw evaluation → summary → statistics → paper-ready derived CSV まで一気通貫実行可能に拡張

## 14. Dependency Reproducibility
- [x] `uv.lock` を生成・配置

## 15. Tests の追加・実行
- [x] 各種ユニットテストを `tests/` に追加（CPU テストと GPU テストの分離）
- [x] 全ユニットテストのパスを確認（75 passed）

## 16. README 更新
- [x] 各 Stage README および root README を論文用 Section 名（§3 Behavioral Characterization, §4 Shared Representation and Causal Overlap, §5 Post-training-Associated Reorganization, §6 From Decodability to Causal Leverage）と整合させて更新

## 17. 完了レポート作成
- [x] 要求された A〜J の項目を網羅した詳細レポートを作成

## 18. 論文主張の査読耐性向上・表現精緻化
- [x] Core Thesis から「単なる出力上の模倣」を削り、査読耐性の高い表現に刷新
- [x] V1 の「shared causal mechanism」を「partially overlapping causally relevant representations and intervention-sensitive sites」へ精緻化
- [x] V2 の解釈を「post-training-associated reorganization」および再現性検証として客観化
- [x] V3 の「acquire causal leverage / only」を「where affect-relevant information exerts measurable causal leverage over self-report」「concentrated at particular layers and generation stages」へ精緻化
- [x] V3 の「utilization」と「causal leverage」を分離（Direction injection: sufficiency/leverage, Subspace removal: necessity/endogenous relevance）
- [x] Behavioral の covariation を「Reader and Self covary in their responses to controlled affective changes」と定義
- [x] 仮説図の結論先取りを解消し、内部表現からの計算分岐として記述
- [x] 729 VAD / 81 VA の Stage 間推論境界（contrast / relative pattern）および Supplementary 感度分析計画を明記
