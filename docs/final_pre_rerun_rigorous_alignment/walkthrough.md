# 本番実験再実行前の最終厳密化 (全15項目) 完了ウォークスルー

## 1. 概要
本作業では、論文の科学的ストーリー（**Behavioral Covariation $\rightarrow$ V1 Partial Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を完全に維持したまま、本番実験再実行前の最終厳密化（全15項目）を実施・検証しました。

全ての修正項目について実装・テスト・検証を完了し、主要 pytest テストスイート **79 passed / 0 failed**、および統合ランナーによる全ステージ（Behavioral, V1, V2, V3）の end-to-end dry-run 実行が完全に通過することを確認しました。また、本番の `results/` ディレクトリは `.gitkeep` のみで一切汚染されておらず、すべてのテスト出力は `dry_run/` 配下に隔離されています。

---

## 2. 実施した修正項目一覧と検証結果

### 【P0: 本番推定値・主主張に直結する重要点】

1. **【P0-1】V1 Phase C の layer indexing 1層ずれ修正**
   - **ファイル**: `v1/primary/run_phase_c.py`, `tests/test_refinement_suite.py`
   - **変更内容**: `extract_hidden_states()` で `get_block_hidden_state` をインポートし、embedding (index 0) を除外して Transformer block 0..L-1 を取得するよう修正。
   - **検証**: `test_phase_c_block_index_mapping` 単体テストを追加し、通過確認。

2. **【P0-2, P0-3】V3 Confirmatory H2 Sufficiency の中立文方向注入への修正 & レイヤー分離**
   - **ファイル**: `v3/primary/run_confirmatory_replication.py`, `configs/v3_experiments.yaml`
   - **変更内容**: 
     - Sufficiency (H2) は中立文 (`prompt_neu_self`) への方向加算注入とし、`shift = injected - clean_neu` を測定。
     - Necessity (H3) は臨床文からの 2D 部分空間除去（減衰測定）を維持。
     - レイヤー解決を介入目的別に分離: `sufficiency_layer = round(suff_rel_depth * (num_layers - 1))` (depth ≈ 0.5), `mediation_layer`, `temporal_map_layer`。
     - `configs/v3_experiments.yaml` に `confirmatory: sufficiency_relative_depth: 0.5` および `spatiotemporal: causal_reference_alpha: 1.0` を追加。

3. **【P0-4】V2 RQ3 に matched-plain causal map を追加**
   - **ファイル**: `v2/primary/run_rq3_causal_map.py`
   - **変更内容**: `model_groups` を `Base plain`, `Instruct matched-plain`, `Instruct native-chat` の3条件に拡張。pair-level CSV の各レコードに `format_condition`（`"plain"`, `"matched_plain"`, `"native_chat"`）を記録保存。

4. **【P0-5】V1 dry-run の出力先ディレクトリ完全分離**
   - **ファイル**: `v1/primary/run_phase_a.py`, `run_phase_b.py`, `phase_c/run_e6_specialization.py`, `phase_c/summarize_phase_c.py`, `src/affective_empathy_eval/run.py`
   - **変更内容**: `--dry-run` 指定時に `args.out_dir = os.path.join(args.out_dir, "dry_run")` へリダイレクト。本番ディレクトリにダミー成果物が混入するのを防止。

---

### 【P1: 統計・推定の頑健性と不整合解消】

5. **【P1-6】V2 Confirmatory H3 を matched-plain Primary に設定**
   - **ファイル**: `v2/primary/run_confirmatory_analysis.py`
   - **変更内容**: `df_pair` から `(alignment=="base" & format_condition=="plain") | (alignment=="inst" & format_condition=="matched_plain")` を Primary LMM として fit。native-chat は Secondary として分離。

6. **【P1-7】V2 Confirmatory H4 も matched-plain recovery AUC を Primary に設定**
   - **ファイル**: `v2/primary/run_confirmatory_analysis.py`
   - **変更内容**:
     - Primary: `auc_recovery_matched_plain`（Self vs Reader 差分）
     - Secondary: `auc_recovery`（native-chat 差分）
     - Mechanistic control: `auc_recovery_aligned`（Procrustes-aligned 差分）
     - Peak: `max_recovery_ratio_matched_plain`

7. **【P1-8】V3 RQ2 の $C(l,t)$ を $\alpha=1.0$ (reference alpha) に固定**
   - **ファイル**: `v3/primary/run_rq2_spatiotemporal_maps.py`
   - **変更内容**: $\alpha$ sweep の末尾ではなく、`causal_reference_alpha = 1.0` のインデックスから $C(l,t)$ を算出。JSON に `"causal_reference_alpha": 1.0` を保存。

8. **【P1-9】V3 RQ1 の attenuation ratio で natural shift < 0.05 を除外**
   - **ファイル**: `v3/primary/run_rq1_state_induction.py`
   - **変更内容**: `natural_shift > 0.05` のサンプルのみ ratio を計算。分母が小さすぎるサンプルを 0.0 として捏造追加しないよう修正。

9. **【P1-10】V3 RQ3 で valid ratio sample が 0件のとき NaN に設定**
   - **ファイル**: `v3/primary/run_rq3_path_mediation.py`, `v3/primary/run_confirmatory_replication.py`
   - **変更内容**: 有効サンプル数が 0 の場合は、0.0 ではなく厳密に `NaN` として出力・報告。

10. **【P1-11】V2 H1a geometry で matched 欠損時に RuntimeError**
    - **ファイル**: `v2/primary/run_confirmatory_analysis.py`
    - **変更内容**: matched-plain の指標（`reader_distortion_matched`, `self_distortion_matched`, `rsa_reader_matched`, `rsa_self_matched`）が存在しない場合、native へのサイレントフォールバックを禁止し `RuntimeError` を送出。

11. **【P1-12】RSA は `rsa_similarity` と明記**
    - **ファイル**: `v2/primary/run_confirmatory_analysis.py`, `v2/primary/run_rq1_rq2_cross_decoding.py`
    - **変更内容**: RSA 指標名を `rsa_similarity` と明記し、相関・類似度としての解釈を統一。

12. **【P1-13】manifest / cache validation の厳格化**
    - **ファイル**: `src/affective_empathy_eval/manifests.py`
    - **変更内容**: config hash, dataset hash, code version, dry-run フラグの完全一致検証を維持・強化。

---

### 【P2: 記述整合・テスト修正】

13. **【P2-14】`tests/run_all_v3_dryruns.py` を production runner 呼び出し形式に改修**
    - **ファイル**: `tests/run_all_v3_dryruns.py`
    - **変更内容**: 旧スクリプトの直接実行から、production primary スクリプト (`v3/primary/run_*.py --dry-run`) を subprocess で実行し、`dry_run/` 配下の成果物を検証するハーネスに刷新。
    - **検証**: 実行し、全4テスト（RQ1, RQ2, RQ3, Confirmatory）の合格を確認。

14. **【P2-15】V3 $\beta$-map の emotion covariate 重複防止**
    - **ファイル**: `src/affective_empathy_eval/data.py`
    - **変更内容**: `v3_stimulus_covariate()` において、`target_emotion` がある場合は `emotion` を除外してダミー変数化し、多重共線性を防止。

---

## 3. テスト及びフルパイプライン検証結果

### 3.1 Python コンパイル検証
```bash
.venv/bin/python -m py_compile \
  v1/primary/run_phase_a.py \
  v1/primary/run_phase_b.py \
  v1/primary/run_phase_c.py \
  v1/primary/phase_c/run_e6_specialization.py \
  v1/primary/phase_c/summarize_phase_c.py \
  v2/primary/run_rq1_rq2_cross_decoding.py \
  v2/primary/run_rq3_causal_map.py \
  v2/primary/run_confirmatory_analysis.py \
  v3/primary/run_rq1_state_induction.py \
  v3/primary/run_rq2_spatiotemporal_maps.py \
  v3/primary/run_rq3_path_mediation.py \
  v3/primary/run_confirmatory_replication.py \
  src/affective_empathy_eval/data.py \
  src/affective_empathy_eval/manifests.py \
  src/affective_empathy_eval/run.py \
  tests/run_all_v3_dryruns.py
```
$\rightarrow$ **エラーなし (Exit Code 0)**

### 3.2 Pytest 全件実行結果
```bash
.venv/bin/python -m pytest tests/ -q
```
$\rightarrow$ **79 passed, 4 warnings in 85.42s (Exit Code 0)**

### 3.3 V3 Dry-run テストハーネス実行結果
```bash
.venv/bin/python tests/run_all_v3_dryruns.py
```
$\rightarrow$ **All V3 dry-run tests successfully passed! (Exit Code 0)**

### 3.4 統合ランナー End-to-End Dry-run 実行結果
```bash
.venv/bin/python -m affective_empathy_eval.run --stage all --dry-run --max-samples 2 --model-set primary_small --force-after-no-go
```
$\rightarrow$ **All requested stages completed successfully! (Exit Code 0)**
- Behavioral, V1 (Phase A, B, C, E6, Summary), V2 (RQ1, RQ2, RQ3, RQ4, Confirmatory), V3 (RQ1, RQ2, RQ3, Confirmatory) がすべて完全自動で実行・完了。

### 3.5 本番 Results ディレクトリの完全性確認
```bash
git status --short
```
$\rightarrow$ `results/` ディレクトリにコミット対象外の未追跡ファイルは一切存在せず、`.gitkeep` のみのクリーン状態が完全に維持されていることを確認。

---

## 4. 結論
以上の通り、本番実験の再実行に向けた全ての科学的・実装的厳密化（全15項目）が完了しました。
論文の構成・ストーリーとコード実装の完全な整合が確保されており、本番フル実行へ安全に進むことが可能です。
