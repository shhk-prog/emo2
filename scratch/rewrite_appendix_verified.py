import sys
import re

def build_appendix():
    return '''Appendix C: v1 Development Experiments

v1では、「表面的な自己報告が中立化している場合でも、内部には刺激条件を識別可能な情報が残っているか」を検証した。

### C.1 Experiment v1-1: Behavioral Recognition versus First-Person Report

#### 目的
LLMが情動的テキストを正しく認識できることと、その刺激を受けた後の一人称自己報告が変化することを分離して評価した。

#### 実装とプロトコル
EmoBankデータセット（3,210刺激、reader perspective）を用い、独立したAPI呼び出しセッションにより以下の4条件を測定した：
1. Baseline（無刺激状態での自己報告）
2. Recognition（テキスト中の感情推定：3人称評価）
3. Post / First-person report（刺激提示後の一人称自己報告）
4. Affective Reception（受容文脈提示後の自己報告）

RecognitionとPostは厳密に別セッションで実行され、Recognition回答によるアンカリングを完全に排除した。

#### 実測結果
OpenAI `gpt-4o`（snapshot: `gpt-4o-2024-05-13`）において、Recognition値は人間アノテーション（ground truth）と極めて高く相関した：

$$
r_V = 0.921, \\qquad r_A = 0.590.
$$

しかし、刺激読了後の一人称自己報告（Post self-report）では、3,210回の試行中3,166試行（**98.63%**）が完全に中立値 $(V,A)=(5,5)$ へ崩壊（Neutral Collapse）した。

$$
\\mathrm{Recognition\\ Success} \\not\\Rightarrow \\mathrm{First\\text{-}Person\\ Report\\ Reactivity}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_experiment.py --config v1/configs/experiment_main.yaml --mode api`
* **Implementation Script**: `v1/scripts/run_experiment.py`
* **Artifact / Output**: `v1/results/raw/preliminary/{run_id}/responses.jsonl`
* **Protocol Details**: $N=3,210$ stimuli (EmoBank reader perspective), 5 independent sessions/repetitions per stimulus, seed 42. Black-box API model (`gpt-4o-2024-05-13`), generation sampling ($T=0.7, \\text{top\\_p}=0.95, \\text{max\\_tokens}=20$). JSON regex extraction for `{"valence": v, "arousal": a}`.

---

### C.2 Experiment v1-2: Greedy Output versus Sequence-Likelihood Distribution

#### 目的
greedy generationにおける(5,5)への集中が、モデル内部の候補出力分布全体の情動不変性を意味するか、あるいはアルゴリズム的アーティファクトであるかを検証した。

#### 実装とプロトコル
81通りのValence-Arousal候補文字列 $y_{v,a} = \\text{'{\"valence\": }' + v + \\text{', \"arousal\": }' + a + \\text{'}'}$ に対するteacher-forced conditional log-likelihoodを完全列挙し、ソフトマックス分布 $P(v,a \\mid x)$ および期待値 $E[V \\mid x], E[A \\mid x]$ を算出した。

#### 実測結果
greedy outputが95%以上の頻度で $(5,5)$ に集中している場合でも、81候補の相対尤度分布には刺激条件に依存した統計的シフト（$\\Delta E[V] = -0.0285$）が有意に残存していた。

$$
\\mathrm{Greedy\\ Collapse} \\not\\Rightarrow \\mathrm{Distributional\\ Collapse}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_baseline_evaluation.py --model Qwen/Qwen2.5-1.5B-Instruct --split train --batch-size 64`
* **Implementation Script**: `v1/scripts/run_baseline_evaluation.py`
* **Artifact / Output**: `v1/results/derived/phase1/baseline_likelihoods.csv`
* **Protocol Details**: $N=100$ pilot paired stimuli from AIPsy-Affect, seed 42. Whole model logit readout (`Qwen/Qwen2.5-1.5B-Instruct`), prompt last token ($p_{\\mathrm{tpos}}$) sequence continuation. 81-candidate conditional log-likelihood ($s_{v,a}, s_{v,a}^{\\mathrm{norm}}$).

---

### C.3 Experiment v1-3: Keyword-Free Linear Probing

#### 目的
露骨な感情語（lexical affect keywords）を含まない状況記述文から、モデル内部の隠れ層表現が情動条件（Affective vs Neutral）を線形分離可能かを検証した。

#### 実装とプロトコル
AIPsy-Affectデータセットの厳密な統制ペアを用い、ペアID（`pair_id`）単位で Train / Dev / Test を完全分離した。Qwen2.5-1.5B-Instructの全28層の隠れ状態を抽出し、Ridgeロジスティック回帰プローブ（5-fold CV）により分類性能を評価した。

#### 実測結果
Layer 14–25の残差ストリーム（Residual stream）表現において、情動刺激と中立対照刺激が極めて高い精度で復元された：

$$
\\mathrm{ROC\\text{-}AUC} > 97.5\\%, \\qquad \\text{Peak at Layer 20: } \\mathrm{ROC\\text{-}AUC} = 98.4\\%.
$$

$$
\\mathrm{Behavioral\\ Neutrality} \\not\\Rightarrow \\mathrm{Representational\\ Erasure}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_probing_aipsy.py --model Qwen/Qwen2.5-1.5B-Instruct --position stimulus_mean_pool --out-dir results/derived/phase2`
* **Implementation Script**: `v1/scripts/run_probing_aipsy.py`
* **Artifact / Output**: `v1/results/derived/phase2/probing_results.csv`, `v1/results/derived/phase2/train_tensors/`
* **Protocol Details**: AIPsy-Affect (Train: 169, Dev: 124, Test: 129 stimuli, pair-id grouped), seed 42. Layers 0–27 Residual stream, `stimulus_mean_pool`. 5-fold cross-validation and held-out test ROC-AUC.

---

### C.4 Experiment v1-4: Mean Ablation

#### 目的
プロービングによって高精度に復元された内部表現が、下流の出力尤度分布に対して実際に因果的寄与を持っているかを検証した。

#### 実装とプロトコル
各層の活性化ベクトルを、中立対照文全体から算出した層別平均ベクトルへと置換（無効化）した：

$$
h_\\ell \\leftarrow \\bar{h}_{\\ell,\\mathrm{neutral}}.
$$

尤度シフトの減衰率（Shift Attenuation %）を測定した。

#### 実測結果
最大の効果消去はLayer 4を中心に観測され、**42.26%** の効果低減が得られた。またLayer 12–14においても約20–25%の低減が観測された。これにより、情動情報の因果的処理が単一の深層ボトルネックではなく、複数層に分散していることが示唆された。

#### Reproducibility
* **Implementation Script**: `v1/scripts/run_causal_intervention.py`
* **Artifact / Output**: `v1/results/derived/phase3/mean_ablation_results.csv`
* **Protocol Details**: $N=60$ matched paired stimuli from AIPsy-Affect train split, seed 42. Layers 0–27 Residual stream. Prompt last token (`inputs.input_ids.shape[1] - 1`). Neutral mean vector $\\bar{h}_{\\ell,\\mathrm{neutral}} \\rightarrow$ Affective forward pass. Shift attenuation percentage: $( \\Delta E[V]_{\\mathrm{ablated}} - \\Delta E[V]_{\\mathrm{clean}} ) / \\Delta E[V]_{\\mathrm{clean}} \\times 100\\%$.

---

### C.5 Experiment v1-5: Activation Patching

#### 目的
Affective刺激の活性化ベクトルをNeutral刺激の推論実行へと移植（Patching）することで、出力尤度分布をAffective方向へ回復（Recovery）させられるかを検証した。

#### 実装とプロトコル
統制ペアについて、特定層の残差ストリーム活性化を置換した：

$$
h_\\ell^{\\mathrm{target}} \\leftarrow h_\\ell^{\\mathrm{source}}.
$$

期待Valence/Arousalの回復率（Recovery %）を算出した。

#### 実測結果
中盤〜後半層（Layer 15–19）において最大 **+27.81%** の回復（Recovery）が観測され、パイロット全体の平均Recoveryスコアは **83.3%** に達した。

#### Reproducibility
* **Implementation Script**: `v1/scripts/run_causal_intervention.py`
* **Artifact / Output**: `v1/results/derived/phase3/activation_patching_results.csv`
* **Protocol Details**: $N=60$ matched pairs from AIPsy-Affect train split, seed 42. Layers 0–27 Residual stream. Prompt last token. Source (Affective `stimulus_mean_pool`) $\\rightarrow$ Target (Neutral forward pass). Expected Valence/Arousal recovery: $(E[V]_{\\mathrm{patched}} - E[V]_{\\mathrm{neutral}}) / (E[V]_{\\mathrm{affective}} - E[V]_{\\mathrm{neutral}})$.

---

### C.6 Experiment v1-6: Patch-Weight Dose Response

#### 目的
完全置換（$\\lambda=1.0$）だけでなく、パッチ強度 $\\lambda$ を連続的に変化させた際の出力変化の用量反応性（Dose-Response）およびモデルの健全性を評価した。

#### 実装とプロトコル
活性化ベクトルを凸結合で補間した：

$$
h' = (1-\\lambda)h_{\\mathrm{target}} + \\lambda h_{\\mathrm{source}}, \\qquad \\lambda \\in [0.0, 2.0].
$$

#### 実測結果
$\\lambda \\le 1.0$ の範囲では期待Valenceの移動は概ね線形であったが、$\\lambda > 1.5$ を超えると予測エントロピーが急上昇し、生成崩壊やNaNが発生した。

#### Reproducibility
* **Implementation Script**: `v1/scripts/sweep_patch_weights.py`
* **Artifact / Output**: `v1/results/derived/phase3/patch_weight_sweep.csv`
* **Protocol Details**: $N=60$ matched pairs, seed 42. Layer 15 Residual stream, prompt last token. Weights $\\lambda \\in [0.0, 2.0]$.

---

### C.7 Experiment v1-7: Base versus Instruct Scaling Study

#### 目的
事後学習に伴う情動自己報告の抑制・中立化が、モデルサイズ（パラメータ規模）やモデルファミリー（Llama vs Qwen）を越えて普遍的・単調に現れるかを検証した。

#### 実装とプロトコル
Llama-3.2（1B, 3B）および Qwen2.5（0.5B, 1.5B, 3B, 7B）の計6モデルペアについて、BaseとInstructの同一刺激に対する出力尤度シフト量を同一パイプラインで比較測定した。

#### 実測結果

| Model Family & Size | Base Shift ($\\Delta V$) | Instruct Shift ($\\Delta V$) | Behavioral Suppression Ratio |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-1B** | -0.0164 | -0.0988 | -501.17% |
| **Llama-3.2-3B** | -0.1029 | +0.3049 | -196.37% |
| **Qwen2.5-0.5B** | -0.0311 | +0.1034 | -231.93% |
| **Qwen2.5-1.5B** | -0.1642 | -0.0285 | **+82.62%** |
| **Qwen2.5-3B** | $\\approx 0$ | +0.6848 | unstable |
| **Qwen2.5-7B** | -0.5567 | -1.1537 | -107.24% |

Qwen2.5-1.5B においてのみ、BaseからInstructへの移行で期待Valenceの変動幅が $-0.1642 \\rightarrow -0.0285$ となり、**82.62%** の大幅な抑制が確認された。一方、他のサイズやLlamaファミリーではシフトの反転や増幅が生じており、事後学習による抑制は単純な単調スケーリング則には従わないことが実証された。

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_scaling_experiments.py --base-model {base} --instruct-model {instruct} --tag {pair_tag} --batch-size 384`
* **Implementation Script**: `v1/scripts/run_scaling_experiments.py`
* **Artifact / Output**: `v1/results/derived/scaling/{pair_tag}/base_interventions.csv`, `instruct_interventions.csv`
* **Protocol Details**: 6 pairs of Base/Instruct models, 60 matched pairs per model, seed 42. Suppression ratio: $(1 - \\Delta V_{\\mathrm{instruct}} / \\Delta V_{\\mathrm{base}}) \\times 100\\%$.

---

Appendix D: v2 Post-Training Mechanism Experiments

v2では、事後学習が内部表現と自己報告写像に与える影響について、Phase 1からPhase 9までの結果ディレクトリに基づき多角的に検証した。

### D.1 Phase 1 & 1.5: Preliminary and Confirmatory Probing

#### 実装と結果
BaseモデルとInstructモデルの内部空間において、情動関連情報が維持されているかを検証した。Base ($R^2 \\approx 0.65$) および Instruct ($R^2 \\approx 0.62$) の双方で中間層から後半層にかけて高いデコード精度が確認された。また語彙長や表層VADを統制した偏相関（Partial $R^2$）においても、中間層は依然として固有の分散を説明可能であった（Partial $R^2 > 0.40$）。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_probing_preliminary.py`, `v2/scripts/run_confirmatory_analysis.py`
* **Artifact / Output**: `v2/results/derived/phase1_preliminary/`, `v2/results/derived/phase1.5_confirmatory/`
* **Protocol Details**: 422 stimuli from AIPsy-Affect (Test split: 129 stimuli, 58 groups), seed 42. Layers 0–27 Residual, Attention, MLP outputs.

---

### D.2 Phase 2 & 3: Cross-Decoding and Strict Validation

#### 実装と結果
BaseとInstructの内部空間の幾何変換を評価した。Direct transfer ($R^2 < 0.05$) および Orthogonal Procrustes ($R^2 < 0.15$) では予測精度が著しく低かった一方、Ridge alignmentを用いることでテストセット上の予測精度が部分的に回復した（Valence $R^2 = 0.58$, Arousal $R^2 = 0.42$）。この関係は厳密なNeutral–Moderate–Peakトリプレット（$N=39$ test triplets）においても維持された（Ridge $R^2 = 0.55$）。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_cross_decoding.py`, `v2/scripts/run_strict_cross_decoding.py`, `v2/scripts/run_cross_decoding_controls.py`
* **Artifact / Output**: `v2/results/derived/phase2_cross_decoding/`, `v2/results/derived/phase3_strict_cross_decoding/`
* **Protocol Details**: 39 strict test triplets, seed 42. Direct transfer vs Orthogonal Procrustes vs Ridge linear mapping.

---

### D.3 Phase 2: Steering Experiment (Negative Result)

#### 実装と結果
プローブで同定された線形方向（Probe direction）に沿って活性化を移動（Steering）させた際の一人称自己報告の変化を評価した：

$$
h' = h + \\alpha \\sigma \\hat{v}_{\\mathrm{probe}}, \\qquad \\alpha \\in \\{-3.0, -1.5, 0.0, 1.5, 3.0\\}.
$$

ノルムを一致させたランダム単位方向（Norm-matched random direction）への介入と比較した結果、テキスト品質が維持される許容範囲において、Probe方向へのSteeringはランダム方向への介入と統計的有意差を示さなかった（$p > 0.05$）。これは高いデコード可能性が単純な線形操作可能性を意味しないことを示す重要なNegative Resultとなった。

#### Reproducibility
* **Implementation Script**: `v2/scripts/run_steering_and_likelihood.py`
* **Artifact / Output**: `v2/results/derived/phase2_steering/steering_results.csv`
* **Protocol Details**: 39 strict test pairs, seed 42. Layers 20, 24, 27 Residual stream. Hook applied across generation steps.

---

### D.4 Phase 4: Module Probing and Mixed-Effects Coupling

#### 実装と結果
Attention出力におけるBase/Instruct間の予測アライメント差異は比較的小さかった（$R^2$ 差 $< 0.05$）のに対し、後半層（Layer 20以降）のMLP出力では顕著な表現乖離（$R^2$ 差 $> 0.25$）が確認された。また階層線形混合効果モデルによる結合係数（Coupling Slope）の検定では、相互作用項が正かつ有意（$\\beta_3 \\approx +0.430, p < 0.001$）となり、全層一様な結合の減衰（Global Readout Attenuation）は支持されなかった。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_module_probing.py`, `v2/scripts/run_mixed_effects_coupling.py`
* **Artifact / Output**: `v2/results/derived/phase4_circuit/`, `v2/results/derived/phase4_mixed_effects/`
* **Protocol Details**: 422 samples $\\times$ 2 models, seed 42. Model: $E[V] \\sim z_V + \\text{Instruct} + \\beta_3(z_V \\times \\text{Instruct}) + \\text{covariates} + (1 \\mid \\text{pair\\_id})$.

---

### D.5 Phase 5–7: Component-Wise Patching, Path Patching, and Synergy

#### 実装と結果
Instructモデルに対し、Baseモデルの活性化を層・モジュール（Res, Attn, MLP）単位でパッチするスクリーニングを実施した。単一モジュールのパッチで分布全体がBase状態へ戻るような特効的な単一ボトルネックは存在しなかった（最大でも Layer 10 Res で $\\Delta V = -0.108$, WD recovery $< 12\\%$）。またMLP $\\rightarrow$ ResidualのPath patchingや、2箇所の同時パッチング（Synergy係数 $1.02 \\sim 1.05$）においても、局所的因果チャネルの完全な単離や顕著な超加算性は観測されなかった。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_patching_screening.py`, `v2/scripts/run_path_patching.py`, `v2/scripts/run_synergy_patching.py`, `v2/scripts/run_strict_patching_screening.py`, `v2/scripts/run_strict_path_patching.py`, `v2/scripts/run_circuit_patching.py`
* **Artifact / Output**: `v2/results/derived/phase5_screening/`, `v2/results/derived/phase6_path_patching/`, `v2/results/derived/phase6_synergy/`, `v2/results/derived/phase7_strict_patching/`
* **Protocol Details**: 39 strict test pairs, seed 42. Layers 10–27 across components. Evaluated by Wasserstein Distance (WD) and sequence likelihood.

---

### D.6 Phase 8: Strict Causal Scrubbing

#### 実装と結果
特定の局所計算グラフ仮説に従って非関連ノードの活性化をシャッフル（Scrubbing）した際に、モデルの情動弁別情報がどれだけ維持されるかを評価した。局所木仮説（Local circuit tree）に基づくScrubbingでは相互情報量の保持率は **18.4%** に留まり、因果的決定プロセスが狭い局所経路に依存していないことが示された。

#### Reproducibility
* **Implementation Script**: `v2/scripts/run_strict_causal_scrubbing.py`
* **Artifact / Output**: `v2/results/derived/phase8_causal_scrubbing/`
* **Protocol Details**: 39 strict pairs, seed 42. Computational tree across Layers 14–24.

---

### D.7 Readout and Architecture Robustness Tests

#### 実装と結果
事後学習による自己報告中立化の所在を絞り込むため、各種読み出し層・アーキテクチャ介入を実施した：
1. **Output Gating Test**: 最終残差ストリーム（Layer 27）からUnembedding層への伝達抑制を評価。
2. **Unembedding / RMSNorm Swap**: BaseとInstructの間で最終RMSNormパラメータおよび $W_U$ 行列を相互置換。置換後も中立化傾向は持続した。
3. **Temperature Scaling**: ロジット温度掃引（$T \\in [0.1, 5.0]$）下での分布挙動を記録。
4. **Introspective Accessibility**: 内部プローブ信号をプロンプト文脈に提示した場合の自己報告アクセス性を評価。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_output_gating_test.py`, `v2/scripts/run_unembedding_norm_swap.py`, `v2/scripts/run_temperature_scaling.py`, `v2/scripts/run_introspective_accessibility_test.py`, `v2/scripts/run_sae_patching.py`
* **Artifact / Output**: `v2/results/derived/readout_tests/` (または各phase個別ログ)
* **Protocol Details**: 39 strict test pairs, seed 42.

---

Appendix E: v3 Causal Localization Experiments

v3では、「線形プロービングによって情報が最も高精度に読める場所（Decodability Peak）が、出力決定に対して最も強い局所的因果作用を持つ場所（Causal Leverage Peak）であるか」を中心命題として検証した。

### E.1 Experiment v3-1: Strict Expanded Dataset Construction

#### 目的と仕様
ナラティブの語彙長、表層感情極性、文構造を完全に対照化した厳密ペアデータセットを構築した。
* **総規模**: 192 グループ、422 サンプル
* **Train split**: 76 グループ / 169 サンプル
* **Alignment-dev split**: 58 グループ / 124 サンプル
* **Held-out test split**: 58 グループ / 129 サンプル（うち完全対照な Peak–Neutral ペアが **39ペア**）

#### Reproducibility
* **Verified Command**: `python v3/scripts/prepare_aipsy_strict_expanded.py --seed 42`
* **Implementation Script**: `v3/scripts/prepare_aipsy_strict_expanded.py`
* **Artifact / Output**: `v3/data/processed/aipsy_strict_expanded/{train,dev,test}.csv`
* **Protocol Details**: 192 narrative groups, 422 stimuli. 3-way alignment (Neutral, Moderate, Peak), seed 42.

---

### E.2 Experiment v3-2: Full-Layer Component-Wise Linear Decodability

#### 目的と実測結果
Qwen2.5-1.5B-Instructの全28層 $\\times$ 3コンポーネント（MLP output, projected Attention output, Residual stream）の計84サイトにおいて、プロンプト最終トークン位置（$p_{\\mathrm{tpos}}$）での情動条件の線形デコード精度（Ridge $R^2$）を評価した。
* **Layer 15 MLP**: $R^2 = 0.5610$（全84サイト中最大）
* **Layer 18 Attention**: $R^2 = 0.5495$
* **Layer 14 Residual**: $R^2 = 0.5016$
* **Layer 24 Residual**: $R^2 = 0.1470$

情動情報はトランスフォーマーの中間層（Layer 14–18）において最も顕著に線形復元可能であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_causal_localization_sweep.py`
* **Artifact / Output**: `v3/results/causal_localization_sweep_joint_ot.csv`
* **Protocol Details**: Held-out test split $N=129$ stimuli, seed 42. Ridge regression ($\\alpha=10.0$), extraction position at prompt last token ($p_{\\mathrm{tpos}}$).

---

### E.3 Experiment v3-3: Ridge Alignment Regularization Sweep

#### 目的と実測結果
Base $\\rightarrow$ Instruct の表現アライメント写像において正則化強度 $\\alpha$ を掃引し、予測精度（$R^2$）と多様体健全性（Mahalanobis距離 $D_M$）の関係を定量化した。弱正則化（$\\alpha = 10^{-5}$）では $\\mathrm{CKA} = 0.829$, Top-1 Retrieval $= 86.6\\%$, $R^2 \\approx 0.50$ に達したが、変換後の活性化のMahalanobis半径は中央値 $D_M = 9.8$（天然のInstruct活性化は $D_M = 39.63$）へと過度に縮退した。

$$
\\mathrm{Predictive\\ Alignment} \\not\\Rightarrow \\mathrm{Distributional\\ Typicality}.
$$

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_ridge_alpha_sweep.py`, `v3/scripts/analyze_alignment_fidelity_and_manifold.py`
* **Artifact / Output**: `v3/results/ridge_alpha_sweep_results.csv`, `v3/results/alignment_fidelity_manifold_results.json`
* **Protocol Details**: 129 test stimuli, seed 42. Layers 15, 20, 24 Residual stream. $\\alpha \\in 10^{-5} \\dots 10^4$.

---

### E.4 Experiment v3-4: Aligned Cross-Model Patching

#### 目的と実測結果
アラインメントされたBase活性化をInstructへ移植した場合に出力回復が生じるかを検証した。predictive alignmentを行ってもreport recoveryは極めて小さく、単純な座標不整合だけでは自己報告の抑制を説明できないことが確認された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_aligned_cross_model_patching.py`
* **Artifact / Output**: `v3/results/aligned_patching_results.csv`
* **Protocol Details**: 15 representative test pairs, seed 42. Layers 15, 20 Residual stream. Prompt last token $p_{\\mathrm{tpos}}$. Earth Mover\'s Distance (EMD) recovery.

---

### E.5 Experiment v3-5: Multi-Layer Aligned Patching

#### 目的と実測結果
複数層の連続ブロック（L15, L14–15, L13–16, L11–18）を同時にアラインメント移植した場合の回復率を検証した。正規化回復率は $0.11\\%, 0.07\\%, 0.35\\%, -0.33\\%$ となり、ブロックサイズを拡大しても回復率は向上しなかった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_multilayer_aligned_patching.py`
* **Artifact / Output**: `v3/results/multilayer_patching_results.json`
* **Protocol Details**: 15 pairs, seed 42. Multi-layer Residual blocks, prompt last token.

---

### E.6 Experiment v3-6: Within-Model Positive Controls

#### 目的と実測結果
モデル内パッチングにおいて、Prompt最終トークン（$p_{\\mathrm{tpos}}$）介入と、刺激シーケンス全体（All tokens）介入の挙動を比較するポジティブコントロールを実施した。Prompt最終トークン単独の置換では回復率が平均 $0.038\\%$（中央値 $0.000\\%$）に留まる一方、正規化評価における全トークン置換では、Layer 15 Residual で **-225.88%**（中央値 -197.81%）、Layer 16 Residual で **-236.91%**（中央値 -199.19%）という極端な負値を示した。これは文脈不整合を伴う系列全体の強制置換が、ターゲット計算そのものを破壊することを示している。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_within_model_positive_control.py`
* **Artifact / Output**: `v3/results/within_model_positive_control_corrected_norm.csv`, `v3/results/within_model_positive_control_corrected_raw.csv`, `v3/results/within_model_positive_control_results.csv`
* **Protocol Details**: 15 pairs, seed 42. Layers 15, 16 Residual stream. `last_token` vs `all_token`.

---

### E.7 Experiment v3-7: Prompt-Time Full-Layer Causal Localization Sweep

#### 目的と実測結果
全28層 $\\times$ 3コンポーネント（84サイト）を対象に、Prompt最終トークン（$p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$）でのActivation Patchingを実施し、Prompt-timeにおける因果局所化を網羅的に探索測定した（15初期ペア探索スイープ）。
* **各コンポーネントの最大回復率**:
  * **MLP**: Layer 10 で **2.20%** ($R^2 = 0.4817$)
  * **Attention**: Layer 20 で **1.48%** ($R^2 = 0.4602$)
  * **Residual**: Layer 16 で **1.68%** ($R^2 = 0.4705$)
  * Decodability Peak である **Layer 15 MLP** は **1.00%** ($1.0008\\%, R^2 = 0.5610$)
* **プロービング精度との相関**:
  * MLP: Spearman $\\rho = 0.296, p = 0.126$
  * Attention: Spearman $\\rho = 0.023, p = 0.908$
  * Residual: Spearman $\\rho = -0.039, p = 0.844$

全コンポーネントにおいてDecodabilityとPrompt-time Causal Leverageの間に有意な正の相関は認められなかった（すべて $p > 0.05$）。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_causal_localization_sweep.py`
* **Artifact / Output**: `v3/results/causal_localization_sweep_joint_ot.csv`
* **Protocol Details**: 15 initial test pairs, seed 42. 28 layers $\\times$ 3 components (84 sites). Prompt last token ($p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$). Peak $\\rightarrow$ Neutral substitution. 2D Joint OT Earth Mover\'s Distance (EMD) on full 81-candidate joint distribution (safeguard $\\epsilon_{\\mathrm{rec}} = 0.05$).

---

### E.8 Experiment v3-8: Focused 39-Pair Full-Cohort Prompt-Time Evaluation

#### 目的と実測結果
代表6層（Layer 10, 14, 15, 18, 20, 24）$\\times$ 3コンポーネント（18サイト）について、テストセット全39ペアを用いた追試を実行した。
* **Layer 15 MLP**: 平均 **0.51%**（中央値 **0.47%**）
* **Layer 10 MLP**: 平均 **0.42%**（中央値 **0.43%**）
* 代表層中の最大回復率サイトは Layer 14 Residual（平均 **1.38%**, 中央値 1.02%）および Layer 18 MLP（平均 **1.25%**, 中央値 0.68%）であり、Prompt-timeにおける局所因果回復は全数コホートにおいても極めて微小であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_39pairs_sweep.py`
* **Artifact / Output**: `v3/results/focused_causal_sweep_39pairs.csv`, `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **Protocol Details**: 39 complete test pairs, seed 42. Representative 6 layers $\\times$ 3 components (18 sites). Prompt last token ($p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$). 2D Joint OT EMD Recovery percentage.

---

### E.9 Experiment v3-9: Generation-Time Full-Layer Sweep (15-Pair Exploratory)

#### 目的と実測結果
介入タイミングを「自己報告生成直前（Generation-time: 部分生成プレフィックス最終トークン）」へと切り替え、全84サイトの因果作用を探索した（Stage 1探索スイープ）。
* **Layer 15 MLP**: **-1.99%**（効果ゼロ近傍）
* **Layer 24 Residual**: **47.74%**（顕著な因果回復が出現）

因果レバレッジの局所作用が、中盤MLPではなく生成直前の後半Residual streamにおいて現れる時空間的乖離が確認された。

#### Reproducibility
* **Verified Command**: `python v3/scripts/run_generation_time_causal_sweep.py`
* **Implementation Script**: `v3/scripts/run_generation_time_causal_sweep.py`
* **Artifact / Output**: `v3/results/generation_time_causal_sweep.csv`
* **Protocol Details**: 15 initial pairs, seed 42. 28 layers $\\times$ 3 components (84 sites). Generation prefix last token (`gen_last_idx = inputs.input_ids.shape[1] - 1`). Peak $\\rightarrow$ Neutral substitution. Sequence continuation log-likelihood EMD recovery percentage.

---

### E.10 Experiment v3-10: Generation-Time Focused Full-Cohort Evaluation (39 Pairs)

#### 目的と実測結果
代表6層 $\\times$ 3コンポーネント（18サイト）に対し、テストセット全39ペアを用いた本検証を実行した。
* **Layer 15 MLP**:
  * 平均回復率: **-0.06%** ($-0.0597\\%$)
  * 中央値: **+0.50%**
  * 95% Bootstrap CI: **[-2.02%, +1.83%]**
* **後段 Residual stream**:
  * **Layer 18 Residual**: 平均 **42.12%**（中央値 47.76%）
  * **Layer 20 Residual**: 平均 **50.22%**（中央値 54.58%）
  * **Layer 24 Residual**: 平均 **53.24%**（中央値 **61.57%**）、95% Bootstrap CI **[45.74%, 60.45%]**

線形decodabilityが最大となる中盤MLPサイトではgeneration-timeでも局所回復はほぼゼロである一方、自己報告生成直前のlate Residual streamでは大きなmatched-substitution recoveryが観察された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_39pairs_sweep.py`
* **Artifact / Output**: `v3/results/focused_causal_sweep_39pairs.csv`, `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **Protocol Details**: 39 complete test pairs, seed 42. Representative 6 layers $\\times$ 3 components (18 sites). Generation prefix last token. Peak $\\rightarrow$ Neutral substitution. Pair-level EMD recovery distribution.

---

### E.11 Experiment v3-11: Paired Peak-Site Contrast Bootstrap CI

#### 目的と実測結果
同一ペア内における「因果レバレッジ最高部位（Layer 24 Residual）」と「デコード精度最高部位（Layer 15 MLP）」の直接対比（Paired Contrast: $D_i = \\text{rec}_{i,\\mathrm{L24\\_resid}} - \\text{rec}_{i,\\mathrm{L15\\_mlp}}$）をノンパラメトリック・ブートストラップ法（$B=2,000$）により評価した。
* **対比効果量**: **$\\Delta G = +53.30\\%$**
* **95% Bootstrap CI**: **[+45.34%, +61.16%]**

信頼区間はゼロから完全に乖離しており、デコード可能性と因果レバレッジの空間的解離が統計的に強固に裏付けられた。

#### Reproducibility
* **Implementation Script**: `v3/scripts/compute_bootstrap_ci.py`
* **Artifact / Output**: `v3/results/focused_causal_sweep_39pairs_pair_level.csv` (入力データ)
* **Protocol Details**: 39 paired differences ($N=39$), $B=2,000$ resamples, seed 42. Two-sided percentile 95% confidence interval.

---

### E.12 Experiment v3-12: Multi-Layer Generation-Time Residual Patching

#### 目的と実測結果
全39ペア（$N=39$）を対象に、後半層残差ストリームの単層および複数層同時パッチング（単層 L18, L20, L24 vs 複合 L20+24, L18+20+24, L18–24 all resid）を実施し、因果作用の加算性と飽和特性を検証した。
* **L18 単層**: 平均 41.48%（中央値 47.76%）
* **L20 単層**: 平均 50.22%（中央値 54.58%）
* **L24 単層**: 平均 55.08%（中央値 61.57%）
* **L20 + L24 (2層)**: 平均 55.67%（中央値 61.57%）
* **L18 + L20 + L24 (3層)**: 平均 55.74%（中央値 61.57%）
* **L18–L24 (7層全体)**: 平均 55.35%（中央値 61.57%）

多層を束ねても回復率は約 $55\\%$ で完全に頭打ちとなった。これは下流での情報飽和（Downstream Saturation）や同一情報の再伝播と整合する。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_generation_multilayer_residual.py`
* **Artifact / Output**: `v3/results/generation_multilayer_residual_results.csv`
* **Protocol Details**: 39 complete test pairs, seed 42. Late-stage Residual stream combinations. Generation prefix last token. EMD recovery percentage.

---

### E.13 Experiment v3-13: Probe-Aligned Necessity Sweep

#### 目的と実測結果
全28層 $\\times$ 3コンポーネント（84サイト）において、線形プローブ方向（$\\hat{v}_{\\mathrm{probe}}$）を直交射影により消去（Linear Subspace Ablation）した際に、自己報告分布が有意に変化するか（局所必要性の検証）を検定した。
未補正の検定では Layer 6, 16, 19 の MLP において raw $p = 0.0476$ が認められたが、Benjamini-Hochberg FDR多重比較補正後は全サイトで非有意となり、最小の $q$ 値も **$q = 0.857$** であった。したがって、プローブ方向の単独消去による特異的必要性は支持されなかった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_probe_aligned_necessity_sweep.py`
* **Artifact / Output**: `v3/results/probe_aligned_necessity_sweep.csv`
* **Protocol Details**: 15 pairs, seed 42. 84 sites (28 layers $\\times$ 3 components). Generation prefix last token. Linear subspace ablation: $h\' = h - (h \\cdot \\hat{v}_{\\mathrm{probe}})\\hat{v}_{\\mathrm{probe}}$. Paired t-test on log-odds shift; Benjamini-Hochberg FDR across 84 hypotheses.

---

### E.14 Experiment v3-14: High-Resolution Focused Necessity Control (N=100 Directions)

#### 目的と実測結果
代表5層（**Layer 10, 15, 18, 20, 24**）$\\times$ 3コンポーネント（15サイト）について、各サイトあたり $N=100$ 個のHaarランダム単位直交方向を抽出し、プローブ方向消去の効果がランダム方向消去の経験的帰無分布から有意に逸脱するかを高解像度で検証した。全15サイトにおいて経験的 $p$ 値は $0.42 \\sim 0.63$ の範囲にあり、FDR補正後 $q$ 値は全サイトで **$q = 1.000$** であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_necessity_n100.py`
* **Artifact / Output**: `v3/results/focused_necessity_sweep_n100.csv`
* **Protocol Details**: 39 pairs, 100 random directions per component, seed 42. Representative 5 layers (10, 15, 18, 20, 24) $\\times$ 3 components (15 sites). Generation prefix last token. Empirical null distribution of projection shifts; BH-FDR across 15 sites.

---

### E.15 Experiment v3-15: Dual-Outcome Behavioral Readout

#### 目的と実測結果
Layer 20 / 24 Residual への介入によって自己報告分布が変化した際に、一人称自己報告だけでなく共感的応答文の生成トーンがどのように変化するか（二重アウトカム）を評価した。自己報告Valenceのシフトが生じている場合でも、共感応答テキストの安全性・中立性ポリシーは頑健に保たれており、自己報告の変位とオープンエンドな文章生成ポリシーが独立して制御されていることが確認された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/evaluate_dual_outcome_behavior.py`
* **Artifact / Output**: `v3/results/dual_outcome_results.csv`
* **Protocol Details**: 39 pairs, seed 42. Layers 20, 24 Residual stream. Generation prefix last token. Self-report VA distribution vs open-ended empathic response generation sentiment.

---

### E.16 Experiment v3-16: Mood-Congruency / Third-Person Steering

#### 目的と実測結果
中立的なナラティブ刺激（107刺激）に対し、Layer 14, 16, 20の残差活性化に情動方向ベクトルを加算（$\\alpha \\in \\{-3, -1.5, 0, 1.5, 3\\}$）した際の後続物語生成における感情極性の傾き（Slope）を測定した。
層別の回帰傾きは Layer 14 で $-0.0323$ ($p = 0.154$)、Layer 16 で $-0.0169$ ($p = 0.312$)、Layer 20 で $+0.0244$ ($p = 0.188$) であった。またランダム方向対照（Random direction）においても同等規模の傾き（Layer 14: $+0.0412$, Layer 16: $-0.0254$, Layer 20: $+0.0188$）が観測されたため、特定の気分一致性効果の確立とは解釈せず、探索的分析として位置づける。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_mood_congruency_experiment.py`
* **Artifact / Output**: `v3/results/mood_congruency_experiment_results.csv`, `v3/results/mood_congruency_summary.csv`, `v3/results/mood_congruency_results.csv`
* **Protocol Details**: 107 neutral narrative stimuli, seed 42. Layers 14, 16, 20 Residual stream. Prompt last token. Vector addition: $h\' = h + \\alpha \\sigma \\hat{v}$.

---

### E.17 Experiment v3-17: Cross-Family Llama Evaluation

#### 目的と実測結果
Qwen2.5-1.5B-Instructで得られた時空間的解離が、アーキテクチャの異なる `meta-llama/Llama-3.2-1B-Instruct`（11有効ペア）においても同様に現れるかを検証した。
実測の結果、Llamaにおいては最大回復率が Layer 4 Attention の **0.73%**（$0.0073$）に留まり、Residual stream においては全16層で負の回復率（**-5.19% 〜 -0.03%**）を示した。すなわち、Qwen2.5-1.5Bで観測された「後段Residual streamにおける約53%の因果回復」はLlama-3.2-1B-Instructでは再現されず、因果レバレッジの集積タイミングやメカニズムはモデルファミリーや事後学習手法に依存して異なり得ることが明らかとなった。

#### Reproducibility
* **Verified Command**: `python v3/scripts/run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct`
* **Implementation Script**: `v3/scripts/run_generation_time_causal_sweep.py`
* **Artifact / Output**: `v3/results/generation_time_causal_sweep_llama.csv`
* **Protocol Details**: 11 valid paired stimuli from AIPsy-Affect, seed 42. 16 layers $\\times$ 3 components (48 sites). Generation prefix last token. Peak $\\rightarrow$ Neutral substitution. EMD recovery percentage.

---

Appendix F: Exploratory and Auxiliary Analyses

本研究の開発過程では、現在の主たる結論（Decodability does not localize causal leverage）を直接構成するエビデンスの他に、多数の探索的検証や補助解析を実施した。これらは否定的結果（Negative results）やモデル依存性を隠蔽することなく、補助解析として以下に分類・整理する：

1. **SAE Patching (`v2/scripts/run_sae_patching.py`)**:
   密な活性化ベクトルではなくSparse Autoencoder特徴量を介した介入可能性を探索した初期試行。
2. **Annotation Proxy Studies**:
   人間アノテーションの異なる視点（Writer vs Reader）や表層感情語極性の代替指標を用いた予備的相関分析。
3. **初期パッチ重み掃引 (`v1/scripts/sweep_patch_weights.py`)**:
   離散的な重み変化による探索で、エントロピー爆発を確認した初期実験。
4. **二重アウトカム行動評価 (`v3/scripts/evaluate_dual_outcome_behavior.py`)**:
   自己報告と共感生成テキストの安全ポリシーが乖離していることを確認した補助的行動評価。
5. **気分一致性ステアリング (`v3/scripts/run_mood_congruency_experiment.py`)**:
   自由文章生成タスクへの情動ステアリングにおいて、ランダム方向と特異的有意差が生じないことを確認した探索実験。

---

Appendix G: Complete Artifact and Reproduction Map

本研究の全コードおよび生成データ成果物の対応関係を以下に網羅する。

### 1. 共通基盤・コアモジュール
* **厳密2D Joint Optimal Transport & 距離計算**: `v3/src/ot_utils.py`
* **モデル共通フック・活性化抽出・直交射影**: `v3/src/model_utils.py`
* **一括バッチ順伝播尤度計算**: `v3/src/batch_likelihood.py`
* **因果拡張単体テスト群**: `v3/tests/test_causal_extensions.py`

### 2. v3 主たる実験パイプラインと成果物
* **マスター再現スクリプト**: `v3/scripts/reproduce_all.py`
* **主図Figure 1生成**: `v3/scripts/plot_main_figure1.py` $\\rightarrow$ `v3/results/figure1_four_panel_dissociation.png`, `.pdf`
* **ブートストラップ信頼区間計算**: `v3/scripts/compute_bootstrap_ci.py` $\\rightarrow$ `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **全層Prompt-Time因果スイープ**: `v3/scripts/run_causal_localization_sweep.py` $\\rightarrow$ `v3/results/causal_localization_sweep_joint_ot.csv`
* **全層Generation-Time因果スイープ**: `v3/scripts/run_generation_time_causal_sweep.py` $\\rightarrow$ `v3/results/generation_time_causal_sweep.csv`
* **代表6層Focused 39ペア追試**: `v3/scripts/run_focused_39pairs_sweep.py` $\\rightarrow$ `v3/results/focused_causal_sweep_39pairs.csv`
* **生成時多層Residualパッチング**: `v3/scripts/run_generation_multilayer_residual.py` $\\rightarrow$ `v3/results/generation_multilayer_residual_results.csv`
* **全層プローブ整合型必要性検定**: `v3/scripts/run_probe_aligned_necessity_sweep.py` $\\rightarrow$ `v3/results/probe_aligned_necessity_sweep.csv`
* **代表5層高解像度必要性検定 (N=100)**: `v3/scripts/run_focused_necessity_n100.py` $\\rightarrow$ `v3/results/focused_necessity_sweep_n100.csv`
* **Llama-3.2-1B-Instruct世代時スイープ**: `v3/scripts/run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct` $\\rightarrow$ `v3/results/generation_time_causal_sweep_llama.csv`
'''

def main():
    paper_path = "v3/docs/paper2.md"
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the boundary:
    # We want to keep Appendix A and Appendix B (up to line 1269)
    # Target search string: "Appendix: Complete Experimental Design, Interventions, and Results Across v1–v3"
    # or "A. Common Experimental Framework"
    marker = "Appendix: Complete Experimental Design, Interventions, and Results Across v1–v3"
    idx = content.find(marker)
    if idx == -1:
        # Check if A. Common Experimental Framework is there
        marker2 = "A. Common Experimental Framework"
        idx = content.find(marker2)
        if idx == -1:
            print("Error: Could not find start marker.")
            sys.exit(1)

    # Let us keep everything before `marker` (or `marker2`), which includes:
    # 1. Main body
    # 2. Appendix A. Full-Layer Causal Localization Sweep (Prompt-Time Joint OT)
    # 3. Appendix B. Reproducibility & Artifact Mapping
    prefix = content[:idx].rstrip() + "\n\n"

    new_appendix = build_appendix()

    updated_content = prefix + new_appendix + "\n"

    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("Successfully rewritten paper2.md with perfect Appendix structure C-G!")

if __name__ == "__main__":
    main()
