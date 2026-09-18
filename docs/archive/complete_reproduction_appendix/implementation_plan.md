# Implementation Plan: v1–v3 完全再現 Supplementary Methods (Appendix) の精緻化

本計画は、`v3/docs/paper2.md` の Appendix（Lines 1270〜2806）に含まれるすべての実験（v1-1〜v1-7, v2 C.1〜C.22, v3-1〜v3-17）に対し、実装コード、設定ファイル、および保存された成果物（CSV, JSON, JSONL）から直接抽出した**8大再現メタデータ**を構造化して埋め込み、学術論文の Supplementary Methods として最高の厳密性を備えた完全再現仕様書へ精緻化することを目的とする。

## 1. 完全再現仕様ブロック（Reproducibility Specification Block）の共通設計

各実験節に、以下の構造化メタデータブロックを統一的に追記・整理する。

```markdown
#### Complete Reproducibility Specification
* **CLI Command**: 実行コマンドライン例（スクリプト名、主要引数、設定ファイルパス）
* **Sample Size ($N$)**: サンプル数（刺激数、対照ペア数、反復回数、プロンプト数など）
* **Random Seed**: 固定シード値（`seed=42` など）
* **Target Layer & Component**: 対象層およびコンポーネント（MLP, Attention, Residual）
* **Token Position**: 表現抽出および介入適用トークン位置（stimulus mean pool, prompt last token $p_{\mathrm{tpos}}$, neutral last token $n_{\mathrm{tpos}}$, generation prefix last token 等）
* **Intervention Direction**: 介入の方向（Peak $\rightarrow$ Neutral, Neutral $\rightarrow$ Peak, Mean ablation, Steering direction 等）
* **Likelihood / Scoring**: 評価スコアリング手法（81候補シーケンス対数尤度、raw vs normalized、2D Joint OT EMD、ROC-AUC、Ridge $R^2$、Wasserstein Distance 等）
* **Artifact & Output Path**: 正確な出力ファイルパス（CSV, JSON, ログ、図表）
```

---

## 2. 対象実験および抽出メタデータ一覧

### Appendix B: Version 1 Experiments (B.1 – B.7)
| 実験 ID | スクリプト / 設定 | $N$ / Seed | 対象層・部品 | トークン位置 / 方向 | 尤度・スコア | 出力ファイル |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B.1** (API測定) | `run_experiment.py` (`experiment_main.yaml`) | $N=3,210$, seed=42 | 全体API (gpt-4o) | 生成サンプリング ($T=0.7$) | JSON抽出, $r_V, r_A$ | `v1/results/raw/preliminary/{run_id}/responses.jsonl` |
| **B.2** (Sequence Likelihood) | `run_baseline_evaluation.py` | $N=100$ pilot, seed=42 | Qwen2.5-1.5B-Instruct | Prompt last token | 81候補 teacher-forced conditional log-likelihood | `v1/results/derived/phase1/baseline_likelihoods.csv` |
| **B.3** (Linear Probing) | `run_probing_aipsy.py` | Train 169, Dev 124, Test 129 | L0–27 Residual | Stimulus mean pool | 5-fold CV & held-out ROC-AUC | `v1/results/derived/phase2/probing_results.csv` |
| **B.4** (Mean Ablation) | `run_causal_intervention.py --mode ablation` | 60 pairs, seed=42 | L0–27 Residual | Prompt last token, Neutral mean vector $\rightarrow$ Affective target | Shift attenuation %, 42.26%消去 (L4) | `v1/results/derived/phase3/mean_ablation_results.csv` |
| **B.5** (Activation Patching) | `run_causal_intervention.py --mode patch` | 60 pairs, seed=42 | L0–27 Residual | Prompt last token, Affective source $\rightarrow$ Neutral target | Recovery %, 最大 +27.81% (L15–19) | `v1/results/derived/phase3/activation_patching_results.csv` |
| **B.6** (Dose Response) | `sweep_patch_weights.py` | 60 pairs, $\alpha \in [0.0, 2.0]$ | L15 Residual | Prompt last token, Affective $\rightarrow$ Neutral | 線形応答傾き, 飽和点 | `v1/results/derived/phase3/patch_weight_sweep.csv` |
| **B.7** (Scaling Study) | `run_scaling_experiments.py` | Base vs Instruct 6モデル対 | L0–$L_{\max}$ | Stimulus mean pool / Prompt last token | Behavioral suppression ratio, Qwen 1.5B 82.62% | `v1/results/derived/scaling/{pair_tag}/` |

### Appendix C: Version 2 Experiments (C.1 – C.22, Phase 1 – 9)
| 実験 ID / Phase | スクリプト / 設定 | $N$ / Seed | 対象層・部品 | 介入 / 比較方向 | 尤度・スコア | 出力ファイル |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C.1** (データ構築) | `prepare_aipsy_strict.py` | 192 groups, 422 stimuli | N/A | Neutral-Moderate-Peak strict triplets | 語彙長・極性統制 | `v2/data/processed/aipsy_strict/` |
| **C.2–C.5** (Phase 1 & 1.5) | `run_probing_preliminary.py`, `run_confirmatory_analysis.py` | Train/Test split | L0–27 Res, Attn, MLP | Base vs Instruct 表現幾何比較 | Ridge $R^2$, ROC-AUC | `v2/results/derived/phase1_preliminary/`, `phase1.5_confirmatory/` |
| **C.6–C.8** (Phase 2) | `run_cross_decoding.py`, `run_steering_and_likelihood.py` | Test pairs | L0–27 | Base $\rightarrow$ Instruct 写像, Steering | Procrustes vs Ridge, Norm-matched random | `v2/results/derived/phase2_cross_decoding/`, `phase2_steering/` |
| **C.9–C.11** (Phase 3 & 3b) | `run_strict_cross_decoding.py`, `run_stress_test_analysis.py` | Held-out test pairs | L10–27 | Strict Cross-decoding, Temperature stress | Test $R^2$, WD (Wasserstein Distance) | `v2/results/derived/phase3_strict_cross_decoding/`, `phase3b_stress_test/` |
| **C.12–C.14** (Phase 4) | `run_module_probing.py`, `run_mixed_effects_coupling.py` | Test pairs | Attn vs MLP vs Resid | Base vs Instruct Coupling Slope $\beta$ | Mixed-effects slope $\beta$, Layer divergence | `v2/results/derived/phase4_circuit/`, `phase4_mixed_effects/` |
| **C.15–C.18** (Phase 5–7) | `run_patching_screening.py`, `run_path_patching.py`, `run_synergy_patching.py` | 39 pairs | L10–27 (mlp, attn, resid) | Base $\rightarrow$ Instruct, Source $\rightarrow$ Target | $\Delta V, \Delta A$, Entropy, Path synergy | `v2/results/derived/phase5_screening/`, `phase6_path_patching/`, `phase7_strict_patching/` |
| **C.19** (Phase 8) | `run_strict_causal_scrubbing.py` | 39 pairs | L14–24 | Tree-path causal scrubbing | Mutual information preservation % | `v2/results/derived/phase8_causal_scrubbing/` |
| **C.20** (Phase 9) | `run_advanced_patching.py` | 39 pairs | Multi-component | Non-linear interaction | Joint WD recovery | `v2/results/derived/phase9_advanced_patching/` |
| **C.21** (Readout Tests) | `run_output_gating_test.py`, `run_unembedding_norm_swap.py`, `run_temperature_scaling.py`, `run_introspective_accessibility_test.py` | 39 pairs | Final Resid, Unembedding | Norm swap, Temperature sweep, Introspection | Output gating ratio, Calibration error | `v2/results/derived/readout_tests/` |
| **C.22** (統合判定) | `run_hypothesis_evaluation.py`, `run_final_summary_report.py` | 全データ | 全層 | H1–H4 競合仮説評価 | 多重証拠重み付けマトリクス | `v2/docs/post_training_readout_experiment/comprehensive_report.md` |

### Appendix D: Version 3 Experiments (D.1 – D.17)
| 実験 ID | スクリプト / 設定 | $N$ / Seed | 対象層・部品 | トークン位置 / 方向 | 尤度・スコア | 出力ファイル |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D.1** (Strict Expanded) | `prepare_aipsy_strict.py` (v3) | 192 groups, 422 stimuli (Test 39 complete pairs) | N/A | 3-way strict alignment | N/A | `v3/data/processed/aipsy_strict_expanded/` |
| **D.2** (Linear Decodability) | `run_causal_localization_sweep.py` | Test $N=129$, seed=42 | 28 layers $\times$ 3 comps (84 sites) | Stimulus mean pool | 5-fold CV & Test $R^2$, L15 MLP $R^2=0.561$, L24 Resid $R^2=0.147$ | `v3/results/causal_localization_sweep_joint_ot.csv` |
| **D.3** (Ridge Alpha Sweep) | `run_ridge_alpha_sweep.py` | Test $N=129$, $\alpha \in 10^{-5}\dots 10^4$ | L10–24 | Stimulus mean pool | Held-out $R^2$, Condition number | `v3/results/ridge_alpha_sweep_results.csv` |
| **D.4** (Aligned Patching) | `run_aligned_cross_model_patching.py` | 15 pairs | L15, L20 Resid | Prompt last token, Base $\rightarrow$ Instruct | EMD / WD recovery | `v3/results/aligned_patching_results.csv` |
| **D.5** (Multilayer Aligned) | `run_multilayer_aligned_patching.py` | 15 pairs | L10–24 multi-layers | Prompt last token | Multi-layer additive recovery | `v3/results/multilayer_patching_results.json` |
| **D.6** (Positive Control) | `run_within_model_positive_control.py` | 15 pairs | L10, 15, 20, 24 | Prompt last token vs all tokens, Peak $\rightarrow$ Neutral | Recovery % (All tokens: 99.6% vs Last: ~0%) | `v3/results/within_model_positive_control_corrected_norm.csv` |
| **D.7** (Prompt-time Sweep) | `run_causal_localization_sweep.py` | 15 pairs, seed=42 | 28 layers $\times$ 3 comps (84 sites) | Prompt last token $p_{\mathrm{tpos}} \rightarrow n_{\mathrm{tpos}}$, Peak $\rightarrow$ Neutral | Joint 2D OT EMD recovery %, max 1.34% (L24 Attn) | `v3/results/causal_localization_sweep_joint_ot.csv` |
| **D.8** (Focused 39p Prompt) | `run_focused_39pairs_sweep.py` | 39 complete test pairs, seed=42 | 6 layers (10, 14, 15, 18, 20, 24) $\times$ 3 comps | Prompt last token $p_{\mathrm{tpos}} \rightarrow n_{\mathrm{tpos}}$, Peak $\rightarrow$ Neutral | Joint 2D OT EMD recovery %, L15 MLP 0.51% (med 0.22%) | `v3/results/focused_causal_sweep_39pairs.csv` |
| **D.9** (Generation 15p Sweep) | `run_generation_time_causal_sweep.py` | 15 pairs, seed=42 | 28 layers $\times$ 3 comps (84 sites) | Generation prefix last token, Peak $\rightarrow$ Neutral | Recovery %, L15 MLP -0.01%, L24 Resid 54.49% | `v3/results/generation_time_causal_sweep.csv` |
| **D.10** (Focused 39p Gen) | `run_focused_39pairs_sweep.py` | 39 complete test pairs, seed=42 | 6 layers $\times$ 3 comps (18 sites) | Generation prefix last token, Peak $\rightarrow$ Neutral | Recovery %, L15 MLP -0.06% [-2.0%, +1.8%], L24 Resid 53.24% [+45.7%, +60.5%] | `v3/results/focused_causal_sweep_39pairs.csv` & `_pair_level.csv` |
| **D.11** (Paired Contrast CI) | `compute_bootstrap_ci.py` | 39 pairs, $N_{\mathrm{boot}}=2000$, seed=42 | L24 Resid vs L15 MLP | Generation-time paired difference | Percentile Bootstrap CI, $\Delta G = +53.30\% \quad [+45.34\%, +61.16\%]$ | Output logged in console & artifact |
| **D.12** (Multilayer Residual) | `run_generation_multilayer_residual.py` | 15 pairs, seed=42 | L18, L20, L24 single & combinations | Generation prefix last token | Recovery %, L24: 55.08%, L20+24: 55.67%, 3-layer: 55.74%, 7-layer: 55.35% | `v3/results/generation_multilayer_residual_results.csv` |
| **D.13** (Probe-Aligned Necessity) | `run_probe_aligned_necessity_sweep.py` | 15 pairs, seed=42 | 28 layers $\times$ 3 comps (84 sites) | Generation prefix last token, Project out probe direction | Log-odds shift, 84 tests BH-FDR $q=1.000$ (全層不成立) | `v3/results/probe_aligned_necessity_sweep.csv` |
| **D.14** (High-Res Necessity) | `run_focused_necessity_n100.py` | 39 pairs, $N=100$ random dirs, seed=42 | 5 layers $\times$ 3 comps (15 sites) | Generation prefix last token, Orthogonal subspace projection | Mean shift, 15 tests BH-FDR $q=1.000$ (全層非有意) | `v3/results/focused_necessity_sweep_n100.csv` |
| **D.15** (Dual-Outcome) | `run_dual_outcome_behavior.py` | 39 pairs | L20, L24 Resid | Generation prefix last token | Self-report VA shift vs Empathic response generation | `v3/results/dual_outcome_results.csv` |
| **D.16** (Mood-Congruency) | `run_mood_congruency_experiment.py` | 107 neutral stimuli, $\alpha \in \{-3, -1.5, 0, 1.5, 3\}$ | L14, 16, 20 Resid | Prompt last token, Activation addition | Story completion sentiment slope | `v3/results/mood_congruency/mood_congruency_results.csv` |
| **D.17** (Llama Replication) | `run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct` | 11 valid pairs, seed=42 | 16 layers $\times$ 3 comps (48 sites) | Generation prefix last token, Peak $\rightarrow$ Neutral | Recovery %, L10 MLP: 0.17%, L14 Resid: 45.41% | `v3/results/generation_time_causal_sweep_llama.csv` |

---

## 3. 実装・検証手順

1. **`v3/docs/paper2.md` の Appendix 全体の改訂**:
   - Appendix B (v1-1 〜 v1-7) に各実験の詳細仕様ブロックを埋め込み、実測数値（98.6% collapse, AUC>97.5%, L4 ablation 42.26%, patching +27.81%, scaling 82.62%）と完全整合させる。
   - Appendix C (v2 C.1 〜 C.22, Phase 1 〜 9) に各実験の詳細仕様ブロックを埋め込み、H1〜H4仮説判定、Slope $\beta$、WD recovery、Module probing、Component patching の実測値を整理する。
   - Appendix D (v3-1 〜 v3-17) に各実験の詳細仕様ブロックを埋め込み、前回の確定ブートストラップ信頼区間（L15 MLP `[-2.0%, +1.8%]`, L24 RESID `[+45.7%, +60.5%]`, Paired Contrast `[+45.34%, +61.16%]`, Prompt-time `0.51%`）と完全整合させる。
   - 数式記法：生の `[` `]` 表示箇所を標準的な LaTeX `$$ ... $$` または `$...$` に統一する。
2. **検証**:
   - `paper2.md` 内の見出し、数値、信頼区間、スクリプト名、出力ファイル名に誤りがないか自動チェック。
   - 各出力 CSV / JSON パスが実在することを確認。
3. **Walkthrough および成果物保存**:
   - `docs/complete_reproduction_appendix/walkthrough.md` を作成し、精緻化された内容を記録。
