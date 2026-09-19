# 本番再実行前の最終厳密化（第2パス）Walkthrough

## 1. 概要
本作業では、論文ストーリー（**Behavioral Covariation $\rightarrow$ V1 Representation / Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を一切変更することなく、本番再実行前の最終厳密化（全19項目）を実施しました。
コード実装、設定ファイル、README群、テストハーネスの整合性を徹底検証し、主要テスト（78 passed, 1 deselected, 0 failed）および全ステージ統一 dry-run の完全成功を確認しました。

---

## 2. 実施した修正項目

### 2.1 【P0: 本番推定値・主主張に直結する重要修正】
1. **V2 RQ3 `post_training_comparison` キー修正** ([`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py))
   - `inst_self` キー欠損による比較生成スキップを解消。
   - `post_training_comparison_matched`（Primary: Base plain vs Instruct matched-plain）および `post_training_comparison_native`（Secondary: Base plain vs Instruct native-chat）の2系列を完全分離して出力・保存。
2. **V2 RQ3 family summary の matched-plain 主体化** ([`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py))
   - Primary: `inst_matched_reader_c_v_peak`, `inst_matched_self_c_v_peak` 等の matched-plain 指標を中心に集約。
   - Secondary: `inst_native_reader_c_v_peak`, `inst_native_self_c_v_peak` を明示。
3. **V2 RQ3 LMM の matched-plain / native-chat 完全分離** ([`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py))
   - Primary LMM（`Base plain` vs `Instruct matched-plain`）と Secondary LMM（`Base plain` vs `Instruct native-chat`）を独立した DataFrame で fit し、出力 JSON に `primary_matched_plain` と `secondary_native_chat` として分離。
4. **V2 Confirmatory H4 で native への fallback を完全削除** ([`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py))
   - Primary metric では `auc_recovery_matched_plain` および `max_recovery_ratio_matched_plain` を必須化。欠損時は native へのサイレントフォールバックを禁止し、明示的な `RuntimeError` を送出。
5. **V3 Confirmatory H4 で temporal-layer local direction を fit** ([`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py))
   - 0.5層で求めた方向ベクトルを 0.65層に移植する座標系不整合を廃止。
   - CV fold の train split 内で `temporal_map_layer`（depth ≈ 0.65）から抽出した表現 `H_temp` に対し、局所的な方向ベクトル `d_temp_v, d_temp_a` および分散スケール `h_std_temp_v, h_std_temp_a` を fit して H4 生成段階介入を実施（データ重複リークゼロ）。

### 2.2 【P1: 統計・推定の頑健性と不整合解消】
6. **V2 RQ4 cross-family summary の matched-primary 化** ([`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py))
   - `primary_matched_plain`、`secondary_native_chat`、`mechanistic_control_aligned`（Procrustes 幾何統制）の3ブロックに構造化してサマリー出力。
7. **Confirmatory レイヤー相対深度の config 化** ([`configs/v3_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml), [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py))
   - `sufficiency_relative_depth: 0.5`, `temporal_relative_depth: 0.65`, `mediation_relative_depth: 0.65` を YAML 設定化し、コード内の固定定数参照を排除。
8. **RQ3 mediator 相対深度の summary 出力** ([`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py))
   - `confirmation_res` およびサマリー JSON に `"mediator_relative_depth"` を追加。
9. **V3 RQ1 Specificity で reference $\alpha=1.0$ の厳格インデックス参照** ([`configs/v3_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml), [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py))
   - `specificity_reference_alpha: 1.0` を設定。末尾インデックス仮定（`[-1]`）を廃止し、`ref_alpha_idx = alpha_grid.index(ref_alpha)` による完全一致参照を実装。
10. **RQ2 `causal_reference_alpha` の完全一致チェック** ([`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py))
    - `alpha_sweep` 内に含まれない場合に `ValueError` を送出。近似 `argmin` による曖昧さを排除し、`alpha_sweep.index(causal_reference_alpha)` で厳格参照。
11. **V3 介入オペレータ表現の完全分離** ([`v3/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/README.md), [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md))
    - 十分性（Sufficiency）の検証: 中立文への加算注入（Additive Injection $h + \alpha \sigma_h \hat{d}$）
    - 必然性・内生的関連性（Necessity / Endogenous Relevance）の検証: 情動文に対する中心化2D直交部分空間除去（Centered Subspace Removal $h - Q Q^T (h - \mu_{\text{neu}})$）
    - 目的と対象文の完全分離を README 上に明記。
12. **V3 「necessity」表現の精緻化** ([`v3/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/README.md), [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md))
    - 主表現を `endogenous relevance` / `necessity-style evidence` に統一。
13. **Manifest / Cache 検証の強化** ([`src/affective_empathy_eval/manifests.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py))
    - `create_run_manifest()` において `config` 辞書から `dataset_path`、`seed`、`model_revision` を自動抽出し、ファイル SHA256 ハッシュ（`dataset_hash`）と設定ハッシュ（`config_hash`）を厳密に計算・記録。
14. **Model revision / Base-Instruct ペアの Manifest 記録**
    - 各 primary スクリプト（V2 RQ3, RQ4, V3 全スクリプト）の manifest に `base_model_id`, `instruct_model_id`, `family_name`, `dataset_path`, `seed` 等を完全記録。
15. **V1 の論文上の位置づけ明記** ([`v1/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/README.md))
    - V1 Primary: Base モデルにおける Reader と Self の共有表現・因果的重複（Primary Focus）
    - V1 Secondary: Instruct モデルにおける 4条件はモデル内再現（Within-Model Replication）
    - 事後学習に伴う再構成の分析は V2 の責務であることを明文化。
16. **V2 H1a の RSA 表現修正** ([`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py))
    - interpretation を `"post-training-associated geometric distortion and representational similarity under matched-plain conditions"` に修正。
17. **Teacher-forced sequence evaluation の明示** ([`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md), [`v3/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/README.md))
    - 生成ドリフトを統制した一貫した評価プロトコルであることを明記。

### 2.3 【P2: 記述整合・テスト修正】
18. **V3 Confirmatory の重複行・空行のクリーンアップ** ([`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py))
19. **Production test の軽量化** ([`tests/test_production_entrypoints.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_production_entrypoints.py), [`pyproject.toml`](file:///mnt/nas/home/hiromi/src/emo2/pyproject.toml))
    - サブプロセス `--help` テストに `@pytest.mark.slow` を付与し、通常テストを 9秒台に高速化。

---

## 3. 検証結果

### 3.1 Python 構文・コンパイル検証
全修正対象スクリプト（17ファイル）の `py_compile` にて構文エラー・インポートエラーがゼロであることを確認。

### 3.2 pytest テストスイート
- 通常テストスイート: **78 passed, 1 deselected in 9.17s**（全通過）
- slow 指定テスト（`test_all_dispatched_commands_argparse_compatibility`）: **1 passed in 79.76s**（全本番スクリプトの引数互換性を完全確認）

### 3.3 V3 全 Primary スクリプト dry-run テストハーネス
- `tests/run_all_v3_dryruns.py`:
  - `run_rq1_state_induction.py --dry-run`: **PASSED**（ゲート判定 GO）
  - `run_rq2_spatiotemporal_maps.py --dry-run`: **PASSED**
  - `run_rq3_path_mediation.py --dry-run`: **PASSED**
  - `run_confirmatory_replication.py --dry-run`: **PASSED**
  - 結果: **All V3 dry-run tests successfully passed!**

### 3.4 統合ランナー（Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3）全ステージ実行
- コマンド: `python -m affective_empathy_eval.run --stage all --dry-run --max-samples 2 --model-set primary_small --force-after-no-go`
- 結果: **All requested stages completed successfully! (exit code 0)**
  - Behavioral: EmoBank, AIPsy, Summary 全通過
  - V1: Phase A, Phase B, Phase C, E6 (4 families x Base/Instruct = 8 conditions) 全通過、Summary 生成完了
  - V2: RQ1/RQ2, RQ3, RQ4 (4 families), Confirmatory LMM 全通過
  - V3: RQ1 (Gate: GO), RQ2, RQ3, Step 7 Confirmatory (3 families) 全通過

### 3.5 原データおよび本番 results の完全性検証
- `git status --short`:
  - 変更はコード、設定、ドキュメントのみ。
  - `results/` ディレクトリ配下に未追跡ファイルや本番結果の上書きは一切なく、クリーンな状態（`.gitkeep` のみ）が厳格に維持されていることを確認。
