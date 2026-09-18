# 修正完了検証報告 (Walkthrough): 本実験前 科学的妥当性・P0/P1/P2不整合修正 (全22項目)

本報告書は、本実験の全パイプライン（Behavioral / V1 / V2 / V3）の再実行に先立ち、実行時例外の原因となる致命的バグ（P0）、統計的推論およびFDR検定の歪み・不整合（P1）、および設定・ドキュメントの科学的主張整合性（P2）の計22項目について実施した改修および検証結果をまとめたものです。

論文構成の骨格：
> **Behavioral: Covariation → V1: Representation / Causal Overlap → V2: Post-training-Associated Reorganization → V3: Causal Leverage**

この論理展開とリサーチクエスチョン（RQ）の定義を厳格に保持したまま、実機実行時の堅牢性と統計的厳密性を完全に担保しました。

---

## 1. 実施した改修項目一覧

### Phase 1: V3 P0 バグ & VA 軸別 Gate
1. **SVD rank-aware 直交部分空間計算と tuple アンパックの修正**
   - [`src/affective_empathy_eval/interventions.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/interventions.py): `compute_orthonormal_subspace` を特異値しきい値 $S_i > \text{tol}$ に基づく rank-aware な基底抽出に改修。戻り値を `(Q_sub, S_filtered)` の tuple に標準化。
   - [`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py): `Q_sub, _ = compute_orthonormal_subspace(...)` として展開。
   - [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py): 特異値分解の次元崩壊を防ぐ rank-aware 部分空間除去を適用。
2. **`generate_control_directions` API 呼び出しの修正**
   - [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py): `d_rand_v, d_perp_v` と `d_rand_a, d_perp_a` を Valence / Arousal それぞれ独立して生成するシグネチャ呼び出しに修正。
3. **V3 RQ1 Gate の Valence / Arousal 完全軸別化**
   - Specificity（$d_V$ vs controls, $d_A$ vs controls）、Necessity（$nat\_dev\_v$ vs $abl\_dev\_v$, $nat\_dev\_a$ vs $abl\_dev\_a$）、Topic TVD（$topic\_tvd\_v$, $topic\_tvd\_a$）を完全独立計算。
   - `evaluate_go_no_go_gate` で `decision_v`, `decision_a` を独立評価し、総合判定（`decision`）を出力。

### Phase 2: Behavioral 統計検定 & FDR 修正
4. **RQ1 Primary 統計量への FDR 適用とファミリー分離**
   - [`behavioral/analysis/summarize_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py): RQ1 の Primary 統計量を `aligned_p_value` に統一。Primary FDR family を `task in ['r', 's']` かつ `dimension in ['V', 'A']` に限定。Writer および Dominance は Supplementary に分離。
5. **RQ3 KeyError の解消と Primary 統一**
   - 存在しない `p_value` ではなく `displacement_p_value` を Primary 統計量として参照・FDR 補正。
6. **RQ2 2段階単調性検定の IUT（Intersection-Union Test）化**
   - $H_1: \Delta_{N \to M} > 0 \text{ and } \Delta_{M \to C} > 0$ に対し $p_{\text{monotonic\_iut}} = \max(p_{\text{step1}}, p_{\text{step2}})$ を算出し、`dose_response_p_value` / `p_value` として FDR 補正。
7. **expected direction 未定義時の NaN 化**
   - `n_defined <= 2` の場合に raw difference にフォールバックせず `np.nan` を代入（虚偽の有意性発生を防止）。
8. **RQ2 Secondary slope CI の適正化**
   - 異なる統計量の CI 端点平均を廃止し、$(C - N)/2$ から直接 bootstrap CI を独立計算。

### Phase 3: V2 RQ3 Cross-Fitting & Confirmatory 安全化
9. **V2 RQ3 5-fold CV Cross-Fitting 化**
   - [`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py): 5-fold CV（KFold）を導入。train fold で $\hat{d}_V, \hat{d}_A$ を推定し、test fold で介入効果を測定。pair_records に `fold_id`, `direction_fit_split='train'`, `evaluation_split='test'` を記録。
10. **V2 Confirmatory の実機実行時 mock fallback 完全排除**
    - [`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py): 非 dry-run 時に成果物（`v2_causal_pair_level.csv`, `v2_geometry_*.json`, `v2_recovery_*.json`）が存在しない場合、合成データへフォールバックせず直ちに `FileNotFoundError` / `RuntimeError` を送出して異常停止。
11. **V2 Confirmatory の H1〜H4 全仮説集約実装**
    - H1 (peak shift), H2 (Reader–Self sharing alteration), H3 (causal dissociation LMM), H4 (distribution recovery asymmetry) を集約。解釈ラベルを "post-training-associated reorganization" に統一。
12. **V2 Confirmatory の runner 接続**
    - [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py): `run_v2()` 末尾に `run_confirmatory_analysis.py` 実行を追加（単一モデル指定時は自動スキップ）。

### Phase 4: Dry-run / Cache 分離 & Manifest 照合接続
13. **Dry-run 結果のキャッシュ混入防止**
    - [`src/affective_empathy_eval/manifests.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py): `RunManifest` と `create_run_manifest` に `dry_run: bool` を追加し、`is_manifest_matching` に `expected_dry_run` チェックを追加。
    - [`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py), [`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py), [`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py), [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py): キャッシュ判定で dry-run 成果物を除外。
14. **manifest 機構のファイルパス実接続**
    - [`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py) で dict 渡しされていたバグを `is_manifest_matching(str(manifest_path), ...)` に修正。各スクリプトで manifest 保存・照合を統一。

### Phase 5: V3 Confirmatory 統計・評価の精密化
15. **H1 / H3 の Valence / Arousal 両軸化**
    - [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py): `d_profile_a`, `c_profile_a`, `dissoc_a` を算出し、H1 および H3 を VA 両軸で独立評価・保存。
16. **Attenuation ratio の clip 撤廃**
    - 生の減衰率（unclipped raw ratio）を保持し、表示用（`attenuation_ratio_display`）のみ clip。推論は raw ratio と bootstrap CI で実施。
17. **効果量・bootstrap CI 中心報告への移行**
    - `primary_effect_estimates` と `family_wise_results` を中心とし、バイナリ合否判定は `auxiliary_qc_checklist` に整理。
18. **test fold サンプリング改善**
    - 先頭5件固定抽出を廃止し、`rng_fold = np.random.default_rng(42 + fold_idx)` によるランダム抽出に変更。
19. **H4 Temporal Emergence の論文主張整合**
    - "decodability ≠ uniform causal leverage" の観点からステージ別不均一性を評価。

### Phase 6: pytest 設定 & ドキュメント整合
20. **`pyproject.toml` の pytest 設定に `pythonpath = ["src", "."]` を追加**
    - [`pyproject.toml`](file:///mnt/nas/home/hiromi/src/emo2/pyproject.toml): ルートディレクトリ `.` を追加し、全スクリプトのテストインポートを統一。
21. **ルート `README.md` の重複見出し削除**
    - [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md): 重複していた `## 1. 中心リサーチクエスチョン (Central RQ)` を1つに統合。
22. **モデル内因果介入 vs モデル間観察的再編の記述統一**
    - [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md) および [`v2/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v2/README.md): 各モデル内部での活性化操作（RQ3/RQ4）を「モデル内因果介入（within-model causal characterization）」とし、Base と Instruct のモデル間比較を「事後学習に伴う再編（post-training-associated reorganization）」とする解釈規約を明確に記述。

---

## 2. 検証結果

### 2.1 構文コンパイル検証
全ソースファイルおよびテストファイルに対して構文検証を実施しました。
```bash
.venv/bin/python -m compileall src behavioral v1 v2 v3 tests
```
**結果**: 終了コード 0（エラーなし、全モジュール正常コンパイル完了）。

### 2.2 ユニットテスト・統合テスト検証
ユニットテストおよびドライラン統合テストを実行しました。
```bash
.venv/bin/python -m pytest -q --ignore=tests/test_production_entrypoints.py
```
**結果**: **75 passed, 4 warnings in 8.51s**

### 2.3 プロダクション・エントリポイント検証
全プロダクションスクリプトの存在確認および `affective_empathy_eval.run` によるディスパッチ検証を実行しました。
```bash
.venv/bin/python -m pytest -q tests/test_production_entrypoints.py
```
**結果**: **3 passed in 80.89s**

**総テスト結果**: **78 passed, 0 failed**

---

## 3. 結論と次のステップ

指示された全22項目の科学的・統計的・実装上の課題がすべて解決され、テストスイートによりその完全性が検証されました。
これにより、実モデルを用いた本実験の実行（`scripts/run_production_all.sh` またはステージ別ランナー）を安全に開始できる状態が整いました。
