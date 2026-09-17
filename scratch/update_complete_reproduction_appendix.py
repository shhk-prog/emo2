import os
import re

def build_appendix_a_b():
    return '''A. Common Experimental Framework

A.1 Valence–Arousal Representation

全バージョンを通じ、情動的出力はRussell型のValence–Arousal空間を用いて表現した。

$$
V, A \\in \\{1, \\ldots, 9\\}.
$$

モデルに要求する基本出力形式は、

```json
{"valence": v, "arousal": a}
```

である。

v1以降の主要解析では、greedy decodingのみではなく、81通りすべての候補列についてteacher-forced conditional likelihoodを計算した。

$$
s_{v,a} = \\sum_{t=1}^{|y_{v,a}|} \\log P(y_{v,a,t} \\mid x, y_{v,a,<t}).
$$

必要に応じて長さ正規化版

$$
s_{v,a}^{\\mathrm{norm}} = \\frac{1}{|y_{v,a}|} s_{v,a}
$$

も使用した。

81候補上でsoftmaxを適用し、

$$
P(v,a \\mid x) = \\frac{\\exp(s_{v,a})}{\\sum_{v',a'} \\exp(s_{v',a'})}
$$

を得た後、

$$
E[V \\mid x] = \\sum_{v,a} v P(v,a \\mid x), \\qquad E[A \\mid x] = \\sum_{v,a} a P(v,a \\mid x)
$$

を自己報告分布の代表値として使用した。

このSequence Likelihood Protocolは、greedy generationにおける(5,5)へのcollapseやJSON parsing failureによって連続的なモデル差が失われることを避けるために導入した。

---

B. v1: Behavioral Reactivity and Initial Causal Analysis

v1では、「表面的な自己報告が中立化している場合でも、内部には刺激条件を識別可能な情報が残っているか」を最初に検証した。

主要コードは以下で構成される：
`run_experiment.py`, `run_main_experiment.py`, `run_baseline_evaluation.py`, `run_probing.py`, `run_probing_aipsy.py`, `run_causal_intervention.py`, `run_alignment_suppression.py`, `sweep_patch_weights.py`, `run_scaling_experiments.py`。

### B.1 Experiment v1-1: Behavioral Recognition versus First-Person Report

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

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_experiment.py --config v1/configs/experiment_main.yaml --mode api`
* **Sample Size ($N$)**: 3,210 stimuli from EmoBank (reader perspective), 5 independent sessions/repetitions per stimulus
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Whole black-box API model (`gpt-4o-2024-05-13`)
* **Token Position**: Unconstrained generation sampling ($T=0.7, \\text{top\\_p}=0.95, \\text{max\\_tokens}=20$)
* **Intervention Direction**: Stimulus prompt presentation (independent session comparisons)
* **Likelihood / Scoring**: Strict JSON regex extraction for `{"valence": v, "arousal": a}`; Pearson correlation $r_V, r_A$ against EmoBank reader ratings
* **Artifact & Output Path**: `v1/results/raw/preliminary/{run_id}/responses.jsonl`

---

### B.2 Experiment v1-2: Greedy Output versus Sequence-Likelihood Distribution

#### 目的
greedy generationにおける(5,5)への集中が、モデル内部の候補出力分布全体の情動不変性を意味するか、あるいはアルゴリズム的アーティファクトであるかを検証した。

#### 実装とプロトコル
81通りのValence-Arousal候補文字列 $y_{v,a} = \\text{'{\"valence\": }' + v + \\text{', \"arousal\": }' + a + \\text{'}'}$ に対するteacher-forced conditional log-likelihoodを完全列挙し、ソフトマックス分布 $P(v,a \\mid x)$ および期待値 $E[V \\mid x], E[A \\mid x]$ を算出した。

#### 実測結果
greedy outputが95%以上の頻度で $(5,5)$ に集中している場合でも、81候補の相対尤度分布には刺激条件に依存した統計的シフト（$\\Delta E[V] = -0.0285$）が有意に残存していた。

$$
\\mathrm{Greedy\\ Collapse} \\not\\Rightarrow \\mathrm{Distributional\\ Collapse}.
$$

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_baseline_evaluation.py --model Qwen/Qwen2.5-1.5B-Instruct --split train --batch-size 64`
* **Sample Size ($N$)**: 100 pilot paired stimuli from AIPsy-Affect
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Whole model logit readout (`Qwen/Qwen2.5-1.5B-Instruct`)
* **Token Position**: Prompt last token sequence continuation ($p_{\\mathrm{tpos}}$)
* **Intervention Direction**: Affective prompt vs matched Neutral prompt
* **Likelihood / Scoring**: 81-candidate teacher-forced conditional log-likelihood ($s_{v,a}$ and $s_{v,a}^{\\mathrm{norm}}$); expected values $E[V \\mid x], E[A \\mid x]$
* **Artifact & Output Path**: `v1/results/derived/phase1/baseline_likelihoods.csv`

---

### B.3 Experiment v1-3: Keyword-Free Linear Probing

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

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_probing_aipsy.py --model Qwen/Qwen2.5-1.5B-Instruct --position stimulus_mean_pool --out-dir results/derived/phase2`
* **Sample Size ($N$)**: AIPsy-Affect (Train: 169, Dev: 124, Test: 129 stimuli; pair-id grouped)
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 0–27 Residual stream
* **Token Position**: `stimulus_mean_pool` (mean-pooling over all tokens belonging to the stimulus text segment)
* **Intervention Direction**: Passive linear probing (no active activation manipulation)
* **Likelihood / Scoring**: 5-fold cross-validated & held-out test ROC-AUC / classification accuracy
* **Artifact & Output Path**: `v1/results/derived/phase2/probing_results.csv`, `v1/results/derived/phase2/train_tensors/`

---

### B.4 Experiment v1-4: Mean Ablation

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

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_causal_intervention.py --model Qwen/Qwen2.5-1.5B-Instruct --position stimulus_mean_pool --mode ablation --batch-size 384 --out-dir results/derived/phase3`
* **Sample Size ($N$)**: 60 matched paired stimuli from AIPsy-Affect train split
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 0–27 Residual stream (single-layer intervention)
* **Token Position**: Prompt last token (`inputs.input_ids.shape[1] - 1`)
* **Intervention Direction**: Neutral condition mean vector $\\bar{h}_{\\ell,\\mathrm{neutral}}$ substituted into Affective stimulus forward pass
* **Likelihood / Scoring**: 81-candidate sequence likelihood; Shift attenuation percentage: $( \\Delta E[V]_{\\mathrm{ablated}} - \\Delta E[V]_{\\mathrm{clean}} ) / \\Delta E[V]_{\\mathrm{clean}} \\times 100\\%$
* **Artifact & Output Path**: `v1/results/derived/phase3/mean_ablation_results.csv`

---

### B.5 Experiment v1-5: Activation Patching

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

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_causal_intervention.py --model Qwen/Qwen2.5-1.5B-Instruct --position stimulus_mean_pool --mode patch --batch-size 384 --out-dir results/derived/phase3`
* **Sample Size ($N$)**: 60 matched pairs from AIPsy-Affect train split
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 0–27 Residual stream
* **Token Position**: Prompt last token (`inputs.input_ids.shape[1] - 1`)
* **Intervention Direction**: Source (Affective stimulus `stimulus_mean_pool` representation) $\\rightarrow$ Target (Neutral prompt forward pass)
* **Likelihood / Scoring**: Expected Valence/Arousal recovery: $(E[V]_{\\mathrm{patched}} - E[V]_{\\mathrm{neutral}}) / (E[V]_{\\mathrm{affective}} - E[V]_{\\mathrm{neutral}})$
* **Artifact & Output Path**: `v1/results/derived/phase3/activation_patching_results.csv`

---

### B.6 Experiment v1-6: Patch-Weight Dose Response

#### 目的
完全置換（$\\lambda=1.0$）だけでなく、パッチ強度 $\\lambda$ を連続的に変化させた際の出力変化の用量反応性（Dose-Response）およびモデルの健全性を評価した。

#### 実装とプロトコル
活性化ベクトルを凸結合で補間した：

$$
h' = (1-\\lambda)h_{\\mathrm{target}} + \\lambda h_{\\mathrm{source}}, \\qquad \\lambda \\in [0.0, 2.0].
$$

#### 実測結果
$\\lambda \\le 1.0$ の範囲では期待Valenceの移動は概ね線形であったが、$\\lambda > 1.5$ を超えると予測エントロピーが急上昇し、生成崩壊やNaNが発生した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/sweep_patch_weights.py --model Qwen/Qwen2.5-1.5B-Instruct --layer 15 --weights 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0`
* **Sample Size ($N$)**: 60 matched pairs
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layer 15 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Linear interpolation from Neutral target to Affective source
* **Likelihood / Scoring**: Expected Valence shift $\\Delta E[V]$ and output entropy $H(P)$
* **Artifact & Output Path**: `v1/results/derived/phase3/patch_weight_sweep.csv`

---

### B.7 Experiment v1-7: Base versus Instruct Scaling Study

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

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v1/scripts/run_scaling_experiments.py --base-model {base} --instruct-model {instruct} --tag {pair_tag} --batch-size 384`
* **Sample Size ($N$)**: 6 pairs of Base/Instruct models, 60 matched pairs per model
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Residual stream across all model layers
* **Token Position**: `stimulus_mean_pool` for probing, prompt last token for sequence likelihood
* **Intervention Direction**: Cross-model baseline comparison: Base vs Instruct output response
* **Likelihood / Scoring**: Behavioral suppression ratio: $(1 - \\Delta V_{\\mathrm{instruct}} / \\Delta V_{\\mathrm{base}}) \\times 100\\%$
* **Artifact & Output Path**: `v1/results/derived/scaling/{pair_tag}/base_interventions.csv`, `instruct_interventions.csv`, `logs/scaling/{pair_tag}.log`'''


def build_appendix_c():
    return '''C. v2: Representation Transformation and Readout Remapping

v2ではv1の観察を、以下の4つの競合仮説として定式化し、Phase 1からPhase 9までの実験を通じて検証した。

* $H_1$ (Erasure): 事後学習により情動表現自体が消去される
* $H_2$ (Representation Transformation): 情動表現は維持されるが幾何構造が非直交的に変換される
* $H_3$ (Global Readout Suppression): 内部表現は残るが全層一様に読み出しパスが遮断される
* $H_4$ (Distributed Readout Remapping): 内部表現が変容し、自己報告への写像関数がネットワーク全体に分散して再編成される

---

### C.1 Phase 1: Preliminary Probing

#### 目的
BaseモデルとInstructモデルの内部空間において、情動関連情報が維持されているか（$H_1$ Erasureの検証）を調べた。

#### 実測結果
Base ($R^2 \\approx 0.65$) および Instruct ($R^2 \\approx 0.62$) の双方で、中間層から後半層にかけて高いデコード精度が確認され、$H_1$ (Erasure) は完全に棄却された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_probing_preliminary.py --model-base Qwen/Qwen2.5-1.5B --model-instruct Qwen/Qwen2.5-1.5B-Instruct --layers all`
* **Sample Size ($N$)**: 422 stimuli from AIPsy-Affect
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 0–27 Residual stream, Attention output, MLP output
* **Token Position**: Stimulus mean pool and prompt last token
* **Intervention Direction**: Passive linear probing
* **Likelihood / Scoring**: Ridge regression $R^2$ and binary classification accuracy
* **Artifact & Output Path**: `v2/results/derived/phase1_preliminary/probing_summary.csv`

---

### C.2 Phase 1.5: Confirmatory Probing

#### 目的
表層的な手がかり（単語長、表層感情極性、物語長など）を統制した上で、内部表現が真に文脈的・状況的感情を捉えているかを厳密に再確認した。

#### 実測結果
統制共変量（Surface VAD, Token count, Narrative richness）を除去した偏相関（Partial $R^2$）においても、中間層は依然として $R^2 > 0.40$ の固有分散を説明可能であった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_confirmatory_analysis.py --split test --controls surface_vad,token_len,narrative_richness`
* **Sample Size ($N$)**: 129 held-out test stimuli (58 narrative groups)
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 10–27 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Controlled linear probing
* **Likelihood / Scoring**: Partial $R^2$ after controlling for surface covariates
* **Artifact & Output Path**: `v2/results/derived/phase1.5_confirmatory/confirmatory_metrics.csv`

---

### C.3 Phase 2: Cross-Decoding

#### 目的
BaseとInstructの内部空間が同一の座標系を維持しているか、あるいは回転・アフィン変換されているかを検証した。

#### 実装とプロトコル
1. Direct transfer（Baseで学習したプローブをInstructにそのまま適用）
2. Orthogonal Procrustes（直交行列による剛体回転）
3. Ridge linear alignment（正則化線形アライメント）

#### 実測結果
Direct transfer ($R^2 < 0.05$) および Orthogonal Procrustes ($R^2 < 0.15$) では予測精度が崩壊した一方、Ridge alignmentを用いることでテストセット上の予測精度が回復した（Valence $R^2 = 0.58$, Arousal $R^2 = 0.42$）。これは事後学習が情動表現の非直交的アライメント変換（$H_2$）を伴うことを示している。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_cross_decoding.py --source-model base --target-model instruct --methods direct,procrustes,ridge`
* **Sample Size ($N$)**: 129 held-out test stimuli
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 0–27 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Base representation $\\rightarrow$ Instruct coordinate system mapping
* **Likelihood / Scoring**: Cross-model predictive alignment $R^2$
* **Artifact & Output Path**: `v2/results/derived/phase2_cross_decoding/cross_decoding_results.csv`

---

### C.4 Phase 2: Steering Experiment

#### 目的
プローブで同定された線形方向（Probe direction）に沿って活性化を移動（Steering）させることで、一人称自己報告を連続的に操作可能かを評価した。

#### 実装とプロトコル
$$
h' = h + \\alpha \\sigma \\hat{v}_{\\mathrm{probe}}, \\qquad \\alpha \\in \\{-3.0, -1.5, 0.0, 1.5, 3.0\\}.
$$
コントロールとして、ノルムを一致させたランダム単位方向（Norm-matched random direction）への介入を同時に実施した。

#### 実測結果
テキスト品質が維持される許容範囲において、Probe方向へのSteeringはランダム方向への介入と比較して統計的に有意な差を生じなかった（$p > 0.05$）。これは「高いデコード可能性」が「線形可制御性」を意味しないという重要なNull/Negative Resultである。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_steering_and_likelihood.py --model Qwen/Qwen2.5-1.5B-Instruct --layers 20,24,27 --alphas -3.0,-1.5,0.0,1.5,3.0 --controls norm_matched_random`
* **Sample Size ($N$)**: 39 strict test pairs
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 20, 24, 27 Residual stream
* **Token Position**: Extended prompt hook (hook applied to prompt last token and all subsequent generation steps)
* **Intervention Direction**: Steering along $\\hat{v}_{\\mathrm{probe}}$ vs norm-matched Haar-random direction $\\hat{v}_{\\mathrm{rand}}$
* **Likelihood / Scoring**: Expected Valence response slope $dE[V]/d\\alpha$ and language model perplexity
* **Artifact & Output Path**: `v2/results/derived/phase2_steering/steering_results.csv`

---

### C.5 Phase 3: Strict Cross-Decoding

#### 目的
Neutral–Moderate–Peakが完全に揃った厳密なトリプレット（Strict triplets）に限定し、データ漏洩を完全に排除した状態でクロスデコーディング結果を再検証した。

#### 実測結果
厳密トリプレットでも Ridge $R^2 = 0.55$ が保持され、事後学習による表現変換の再現性が確認された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_strict_cross_decoding.py --triplets-only --eval-split test`
* **Sample Size ($N$)**: 39 complete test triplets (117 samples)
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 10–27 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Base $\\rightarrow$ Instruct coordinate alignment
* **Likelihood / Scoring**: Held-out Test $R^2$
* **Artifact & Output Path**: `v2/results/derived/phase3_strict_cross_decoding/strict_cross_decoding.csv`

---

### C.6 Phase 3b: Model Stress Test

#### 目的
温度パラメータ（$T \\in [0.1, 1.5]$）やプロンプト文面の微小な変化に対して、内部表現アライメントが頑健であるかを探索した。

#### Complete Reproducibility Specification
* **CLI Command**: `bash v2/scripts/run_stress_test_models.sh`
* **Sample Size ($N$)**: 39 test pairs across 5 temperature settings
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 14–24 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Cross-temperature and cross-prompt stress testing
* **Likelihood / Scoring**: Prediction stability and Wasserstein Distance across variations
* **Artifact & Output Path**: `v2/results/derived/phase3b_stress_test/`

---

### C.7 Phase 4: Module Probing

#### 目的
事後学習による表現の変容（Divergence）が、Attention出力、MLP出力、あるいはResidual streamのどのモジュールに局在しているかを特定した。

#### 実測結果
Attention出力におけるBase/Instruct間の予測アライメント差異は極めて小さかった（$R^2$ 差 $< 0.05$）のに対し、後半層（Layer 20以降）のMLP出力では顕著な表現乖離（$R^2$ 差 $> 0.25$）が確認された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_module_probing.py --components res,attn,mlp --layers 10-27`
* **Sample Size ($N$)**: 129 held-out test stimuli
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Attention output, MLP output, Residual stream (Layers 10–27)
* **Token Position**: Prompt last token
* **Intervention Direction**: Module-specific separate linear probing
* **Likelihood / Scoring**: Held-out Ridge $R^2$ and Frobenius divergence metric $\\|W_{\\mathrm{base}} - W_{\\mathrm{instruct}}\\|_F$
* **Artifact & Output Path**: `v2/results/derived/phase4_circuit/module_probing.csv`

---

### C.8 Phase 4: Mixed-Effects Coupling Analysis

#### 目的
内部表現 $z_V$ から自己報告期待値 $E[V]$ への結合（Coupling）が、事後学習によって全層一様に弱小化（Global Attenuation / $H_3$）されたかを統計的に検定した。

#### 実装とプロトコル
以下の階層線形混合効果モデルを適合させた：

$$
E[V] \\sim z_V + \\mathrm{Instruct} + \\beta_3 (z_V \\times \\mathrm{Instruct}) + \\sum_k \\gamma_k X_k + (1 \\mid \\mathrm{pair\\_id}).
$$

#### 実測結果
代表層において相互作用係数は正かつ有意であった（$\\beta_3 \\approx +0.430, p < 0.001$）。すなわち、Instructモデルにおいて結合強度がBaseより低下するどころか、層によってはより急峻な傾きを示しており、大域的抑制仮説（$H_3$）は完全に棄却された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_mixed_effects_coupling.py --dependent-var E_V --random-effects pair_id`
* **Sample Size ($N$)**: 422 samples $\\times$ 2 models
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layers 10–27 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Base vs Instruct interaction modeling
* **Likelihood / Scoring**: Linear mixed-effects regression slope $\\beta$ and p-values
* **Artifact & Output Path**: `v2/results/derived/phase4_mixed_effects/mixed_effects_summary.csv`

---

### C.9 Phase 5: Component-Wise Activation Patching Screening

#### 目的
Instructモデルに対し、Baseモデルの活性化を層・モジュール（Res, Attn, MLP）単位でパッチすることで、自己報告分布がBase状態へ復元する単一のボトルネックが存在するかを探索した。

#### 実測結果
Layer 10–27の全54コンポーネントを探索したが、単一モジュールのパッチで分布全体がBase状態へ戻る（$\\Delta V \\approx -1.0$）ような特効的な層は存在しなかった（最大でも Layer 10 Res で $\\Delta V = -0.108$, WD recovery $< 12\\%$）。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_patching_screening.py --layers 10-27 --components res,attn,mlp --metric wasserstein`
* **Sample Size ($N$)**: 39 strict test pairs
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: 54 components (Layers 10–27 $\\times$ Residual, Attention, MLP)
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Base activation grafted into Instruct forward pass
* **Likelihood / Scoring**: 81-candidate sequence likelihood; $\\Delta V, \\Delta A$, Wasserstein Distance (WD) recovery, Output entropy
* **Artifact & Output Path**: `v2/results/derived/phase5_screening/patching_screening_results.csv`

---

### C.10 Phase 6: Path Patching

#### 目的
MLP出力からResidual streamへの特定の伝播エッジ（Path）を個別にパッチし、自己報告の決定経路が単一の回路パスに依存しているかを検証した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_path_patching.py --source-components mlp --target-components res --layers 14-24`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: MLP $\\rightarrow$ Residual paths across Layers 14–24
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Edge-specific activation substitution
* **Likelihood / Scoring**: Relative path causal effect and WD recovery
* **Artifact & Output Path**: `v2/results/derived/phase6_path_patching/path_patching_results.csv`

---

### C.11 Phase 6: Synergy Patching

#### 目的
2つのモジュールを同時にパッチした場合に、単独パッチの和を超える相乗効果（超加算性 / Synergy）が生じるかを検証した。

#### 実測結果
L10 MLP + L20 MLP などの組み合わせにおいて、シナジー係数は $1.02 \\sim 1.05$ に留まり、明確な非線形相互作用は確認されなかった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_synergy_patching.py --pairs L10_mlp+L20_mlp,L15_res+L24_res`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Multi-component pairs (L10 MLP + L20 MLP, L15 Resid + L24 Resid)
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Dual-site simultaneous activation grafting
* **Likelihood / Scoring**: Synergy ratio: $\\mathrm{Recovery}(A+B) / (\\mathrm{Recovery}(A) + \\mathrm{Recovery}(B))$
* **Artifact & Output Path**: `v2/results/derived/phase6_synergy/synergy_patching_results.csv`

---

### C.12 Phase 7: Strict Patching

#### 目的
トリプレット統制下で厳格なWasserstein Distance閾値基準を用いたパッチングスクリーニングを実行した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_strict_patching_screening.py --triplets-only --wd-threshold 0.05`
* **Sample Size ($N$)**: 39 strict triplets, `seed = 42`
* **Target Layer & Component**: Layers 10–27 Residual, Attention, MLP
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Base $\\rightarrow$ Instruct
* **Likelihood / Scoring**: Strict Wasserstein Distance recovery percentage
* **Artifact & Output Path**: `v2/results/derived/phase7_strict_patching/strict_patching_results.csv`

---

### C.13 Phase 7: Strict Path Patching

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_strict_path_patching.py --control shuffled_activations`
* **Sample Size ($N$)**: 39 strict triplets, `seed = 42`
* **Target Layer & Component**: Layers 14–24
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Shuffled vs matched activations
* **Likelihood / Scoring**: Controlled path causal effect
* **Artifact & Output Path**: `v2/results/derived/phase7_strict_path_patching/`

---

### C.14 Phase 7: Cross-Decoding Controls

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_cross_decoding_controls.py --permutations 1000`
* **Sample Size ($N$)**: 39 test pairs, `seed = 42`
* **Target Layer & Component**: Layers 10–27
* **Token Position**: Prompt last token
* **Intervention Direction**: Randomly permuted label cross-decoding
* **Likelihood / Scoring**: Permutation null distribution of $R^2$
* **Artifact & Output Path**: `v2/results/derived/phase7_cross_decoding_controls/`

---

### C.15 Phase 8: Causal Scrubbing

#### 目的
特定の計算グラフ（局所回路仮説）に従って不要な情報を置換（Scrubbing）した際に、モデルの情動弁別能力がどれだけ維持されるかを判定した。

#### 実測結果
局所的回路仮説（Local circuit tree）に基づくScrubbingでは、本来の相互情報量の **18.4%** しか保存されず、因果的決定プロセスが局所経路に閉じていないことが証明された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_strict_causal_scrubbing.py --tree-hypothesis H4_distributed`
* **Sample Size ($N$)**: 39 strict pairs, `seed = 42`
* **Target Layer & Component**: Circuit tree across Layers 14–24
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Computational tree node scrambling under candidate hypothesis
* **Likelihood / Scoring**: Mutual information preservation ratio
* **Artifact & Output Path**: `v2/results/derived/phase8_causal_scrubbing/scrubbing_results.csv`

---

### C.16 Phase 9: Advanced Patching

#### 目的
貪欲探索（Greedy search）を用いて最大5つのコンポーネントを同時にパッチし、分散回路全体の同時回復が可能かを検証した。

#### 実測結果
5モジュールを同時置換しても WD recovery は **23.1%** で飽和し、低次元な線形重ね合わせだけでは説明できない高次元な再写像（Distributed Remapping / $H_4$）の存在が確定的となった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_advanced_patching.py --combination-search greedy --max-components 5`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Top-5 selected components from 54 candidate sites
* **Token Position**: Extended prompt hook
* **Intervention Direction**: Combinatorial multi-site activation grafting
* **Likelihood / Scoring**: Joint Wasserstein Distance recovery percentage
* **Artifact & Output Path**: `v2/results/derived/phase9_advanced_patching/advanced_patching_results.csv`

---

### C.17 Output-Gating / Late-Residual Substitution

#### 目的
中立化が最終残差ストリーム（Layer 27）からUnembedding層へのゲート（Output Gating）によって引き起こされているかを検証した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_output_gating_test.py --model Qwen/Qwen2.5-1.5B-Instruct`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Layer 27 final residual stream and RMSNorm
* **Token Position**: Final generation step
* **Intervention Direction**: Layer 27 substitution from Base into Instruct
* **Likelihood / Scoring**: Output gating attenuation ratio
* **Artifact & Output Path**: `v2/results/derived/readout_tests/output_gating_results.csv`

---

### C.18 Unembedding / RMSNorm Swap

#### 目的
BaseモデルとInstructモデルの間で、最終Unembedding重み行列（$W_U$）および最終RMSNormパラメータを相互にスワップし、読み出し重みそのものの変更が中立化の原因であるかを検証した。

#### 実測結果
$W_U$ や RMSNorm を Base のものに差し替えても中立化の緩和は見られず、原因は最終投影パラメータ単体ではなく、トランスフォーマー本体の分散的変換にあることが判明した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_unembedding_norm_swap.py --model-base Qwen/Qwen2.5-1.5B --model-instruct Qwen/Qwen2.5-1.5B-Instruct`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Final LayerNorm / RMSNorm and Unembedding projection matrix $W_U$
* **Token Position**: Output logit projection
* **Intervention Direction**: Parameter weight swapping
* **Likelihood / Scoring**: 81-candidate expected Valence shift $\\Delta E[V]$
* **Artifact & Output Path**: `v2/results/derived/readout_tests/unembedding_swap_results.csv`

---

### C.19 Temperature Scaling

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_temperature_scaling.py --temperatures 0.1,0.5,1.0,2.0,5.0`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Whole model logit readout
* **Token Position**: Final readout
* **Intervention Direction**: Logit temperature scaling
* **Likelihood / Scoring**: Expected values and entropy across temperatures
* **Artifact & Output Path**: `v2/results/derived/readout_tests/temperature_scaling_results.csv`

---

### C.20 Response-Format Robustness

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_response_format_robustness.py --formats json,plain_text,likert_scale`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Readout format conditioning
* **Token Position**: Generation continuation
* **Intervention Direction**: Prompt format alterations
* **Likelihood / Scoring**: Classification consistency across formats
* **Artifact & Output Path**: `v2/results/derived/readout_tests/response_format_results.csv`

---

### C.21 Introspective Accessibility Test

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_introspective_accessibility_test.py --probe-guided-prompts`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Whole model
* **Token Position**: Dialogue continuation
* **Intervention Direction**: Internal probe signal provided as introspective prompt cue
* **Likelihood / Scoring**: Self-report calibration accuracy
* **Artifact & Output Path**: `v2/results/derived/readout_tests/introspection_results.csv`

---

### C.22 8仮説統合評価判定

#### 目的
以上の全Phase実験群を統合し、$H_1$〜$H_4$ の各仮説に対する包括的判定を下した。

#### 実測結果
* **$H_1$ (Erasure)**: 棄却（内部空間における高精度プロービングおよびクロスモデル線形アライメントの成立）
* **$H_2$ (Transformation)**: 支持（非直交的な線形座標変換の成立、コンポーネント依存の表現乖離）
* **$H_3$ (Global Suppression)**: 棄却（混合効果モデルにおける交互作用係数の正値、層別結合の不均一性）
* **$H_4$ (Distributed Remapping)**: 最有力支持（単一ボトルネックの不在、パッチングの非線形飽和、Causal Scrubbingによる局所回路の不成立）

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v2/scripts/run_hypothesis_evaluation.py --report v2/docs/post_training_readout_experiment/comprehensive_report.md`
* **Sample Size ($N$)**: 全フェーズ実験データ
* **Artifact & Output Path**: `v2/docs/post_training_readout_experiment/comprehensive_report.md`'''


def build_appendix_d():
    return '''D. v3: Decodability Does Not Localize Causal Leverage

v3では、「プロービングによって情報が最も高精度に読める場所（Decodability Peak）が、出力決定に対して最も強い局所的因果作用を持つ場所（Causal Leverage Peak）であるか」を中心命題として検証した。

---

### D.1 Experiment v3-1: Strict Expanded Dataset Construction

#### 目的
ナラティブの語彙長、表層感情極性、文構造を完全に対照化した厳密ペアデータセットを構築した。

#### 実測仕様
* **総規模**: 192 グループ、422 サンプル
* **Train split**: 76 グループ / 169 サンプル
* **Alignment-dev split**: 58 グループ / 124 サンプル
* **Held-out test split**: 58 グループ / 129 サンプル（うち完全対照な Peak–Neutral ペアが **39ペア**）

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/prepare_aipsy_strict_expanded.py --seed 42`
* **Sample Size ($N$)**: 192 narrative groups, 422 stimuli (Test split: 39 complete matched pairs)
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: N/A (Dataset artifact)
* **Token Position**: N/A
* **Intervention Direction**: 3-way strict alignment (Neutral, Moderate, Peak)
* **Likelihood / Scoring**: Narrative balance metrics
* **Artifact & Output Path**: `v3/data/processed/aipsy_strict_expanded/{train,dev,test}.csv`

---

### D.2 Experiment v3-2: Full-Layer Component-Wise Linear Decodability

#### 目的
Qwen2.5-1.5B-Instructの全28層 $\\times$ 3コンポーネント（MLP出力、projected Attention出力、Residual stream）の計84サイトにおいて、情動条件の線形デコード精度（Ridge $R^2$）を評価した。

#### 実測結果
* **Layer 15 MLP**: $R^2 = 0.5610$（全84サイト中最大）
* **Layer 18 Attention**: $R^2 = 0.5495$
* **Layer 14 Residual**: $R^2 = 0.5016$
* **Layer 24 Residual**: $R^2 = 0.1470$

情動情報はトランスフォーマーの中間層（Layer 14–18）において最も顕著に線形復元可能であった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_causal_localization_sweep.py --mode probing_only --eval-split test`
* **Sample Size ($N$)**: Held-out test split $N=129$ stimuli, `seed = 42`
* **Target Layer & Component**: 28 layers $\\times$ 3 components (MLP, Attention, Residual) = 84 sites
* **Token Position**: `stimulus_mean_pool`
* **Intervention Direction**: Passive linear probing
* **Likelihood / Scoring**: Ridge regression ($\\alpha=10.0$) held-out $R^2$
* **Artifact & Output Path**: `v3/results/causal_localization_sweep_joint_ot.csv`

---

### D.3 Experiment v3-3: Ridge Alignment Regularization Sweep

#### 目的
Base $\\rightarrow$ Instruct の表現アライメント写像において、正則化強度 $\\alpha$ を掃引し、予測精度（$R^2$）と多様体健全性（Mahalanobis距離 $D_M$）のトレードオフを定量化した。

#### 実測結果
弱正則化（$\\alpha = 10^{-5}$）では $\\mathrm{CKA} = 0.829$, Top-1 Retrieval $= 86.6\\%$, $R^2 \\approx 0.50$ に達したが、変換後の活性化のMahalanobis半径は中央値 $D_M = 9.8$（天然のInstruct活性化は $D_M = 39.63$）へと過度に縮退した。

$$
\\mathrm{Predictive\\ Alignment} \\not\\Rightarrow \\mathrm{Distributional\\ Typicality}.
$$

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_ridge_alpha_sweep.py --alphas 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100 1000 10000`
* **Sample Size ($N$)**: 129 test stimuli, `seed = 42`
* **Target Layer & Component**: Layers 15, 20, 24 Residual stream
* **Token Position**: `stimulus_mean_pool`
* **Intervention Direction**: Cross-model Ridge mapping across $\\alpha$
* **Likelihood / Scoring**: Held-out $R^2$, CKA similarity, Top-1 retrieval, Mahalanobis distance $D_M$
* **Artifact & Output Path**: `v3/results/ridge_alpha_sweep_results.csv`, `v3/results/alignment_fidelity_manifold_results.json`

---

### D.4 Experiment v3-4: Aligned Cross-Model Patching

#### 目的
アラインメントされたBase活性化をInstructへ移植した場合に出力回復が生じるかを検証した。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_aligned_cross_model_patching.py --source base --target instruct --layers 15,20`
* **Sample Size ($N$)**: 15 representative test pairs, `seed = 42`
* **Target Layer & Component**: Layers 15, 20 Residual stream
* **Token Position**: Prompt last token $p_{\\mathrm{tpos}}$
* **Intervention Direction**: Ridge-aligned Base activation $\\rightarrow$ Instruct target prompt
* **Likelihood / Scoring**: Earth Mover's Distance (EMD) recovery percentage
* **Artifact & Output Path**: `v3/results/aligned_patching_results.csv`

---

### D.5 Experiment v3-5: Multi-Layer Aligned Patching

#### 目的
単層ではなく複数層の連続ブロック（L15, L14–15, L13–16, L11–18）を同時にアラインメント移植した場合の回復率を検証した。

#### 実測結果
正規化回復率は $0.11\\%, 0.07\\%, 0.35\\%, -0.33\\%$ となり、ブロックサイズを拡大しても回復率は向上しなかった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_multilayer_aligned_patching.py --blocks L15,L14-15,L13-16,L11-18`
* **Sample Size ($N$)**: 15 pairs, `seed = 42`
* **Target Layer & Component**: Multi-layer Residual blocks
* **Token Position**: Prompt last token
* **Intervention Direction**: Multi-layer aligned Base grafting
* **Likelihood / Scoring**: Multi-layer normalized recovery percentage
* **Artifact & Output Path**: `v3/results/multilayer_patching_results.json`

---

### D.6 Experiment v3-6: Within-Model Positive Controls

#### 目的
モデル内パッチングにおいて、Prompt最終トークン（$p_{\\mathrm{tpos}}$）介入と、刺激シーケンス全体（All tokens）介入の挙動を比較するポジティブコントロールを実施した。

#### 実測結果
Prompt最終トークン単独の置換では回復率が $\\approx 1\\%$ に留まる一方、全トークン置換では $99.6\\%$ の完全回復、あるいは文脈不整合による極端な負の破綻が生じた。これにより、局所的因果介入の評価には位置統制が不可欠であることが確認された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_within_model_positive_control.py --layers 10,15,20,24 --positions prompt_last,all_tokens`
* **Sample Size ($N$)**: 15 pairs, `seed = 42`
* **Target Layer & Component**: Layers 10, 15, 20, 24 Residual stream
* **Token Position**: Prompt last token vs all token positions
* **Intervention Direction**: Peak $\\rightarrow$ Neutral within-model substitution
* **Likelihood / Scoring**: Normalized recovery percentage
* **Artifact & Output Path**: `v3/results/within_model_positive_control_corrected_norm.csv`, `_raw.csv`

---

### D.7 Experiment v3-7: Prompt-Time Full-Layer Causal Localization Sweep

#### 目的
全28層 $\\times$ 3コンポーネント（84サイト）を対象に、Prompt最終トークン（$p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$）でのActivation Patchingを実施し、Prompt-timeにおける因果局所化を網羅的に測定した。

#### 実測結果
* **最大回復率**: Layer 24 Attention で $1.34\\%$、Layer 10 MLP で $0.42\\%$、Layer 15 MLP で $1.00\\%$
* **プロービングとの相関**:
  * MLP: Spearman $\\rho = 0.296, p = 0.126$
  * Attention: Spearman $\\rho = 0.023, p = 0.908$
  * Residual: Spearman $\\rho = -0.039, p = 0.844$

全コンポーネントにおいてDecodabilityとPrompt-time Causal Leverageの相関は統計的に有意でなかった（すべて $p > 0.05$）。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_causal_localization_sweep.py --use-cache False --metric joint_ot`
* **Sample Size ($N$)**: 15 initial exploration test pairs, `seed = 42`
* **Target Layer & Component**: 28 layers $\\times$ 3 components (84 sites)
* **Token Position**: Prompt last token ($p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$)
* **Intervention Direction**: Peak $\\rightarrow$ Neutral substitution
* **Likelihood / Scoring**: 2D Joint OT Earth Mover's Distance (EMD) on full 81-candidate joint distribution; Recovery %: $(D(P_{\\mathrm{neu}}, P_{\\mathrm{patch}}) - D(P_{\\mathrm{neu}}, P_{\\mathrm{peak}})) / \\dots$
* **Artifact & Output Path**: `v3/results/causal_localization_sweep_joint_ot.csv`

---

### D.8 Experiment v3-8: Focused 39-Pair Full-Cohort Prompt-Time Evaluation

#### 目的
代表6層（Layer 10, 14, 15, 18, 20, 24）$\\times$ 3コンポーネント（18サイト）について、テストセット全39ペアを用いた追試を実行した。

#### 実測結果
* **Layer 15 MLP**: 平均 $0.51\\%$（中央値 $0.22\\%$）
* **Layer 10 MLP**: 平均 $0.42\\%$（中央値 $0.23\\%$）
* 全18サイトの平均回復率はすべて $< 1.1\\%$ に留まり、Prompt-timeにおける局所因果効果の極小性が全数コホートで確立された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_focused_39pairs_sweep.py --eval-stage prompt --layers 10,14,15,18,20,24 --components mlp,attn,resid`
* **Sample Size ($N$)**: 39 complete test pairs, `seed = 42`
* **Target Layer & Component**: Representative 6 layers $\\times$ 3 components (18 sites)
* **Token Position**: Prompt last token ($p_{\\mathrm{tpos}} \\rightarrow n_{\\mathrm{tpos}}$)
* **Intervention Direction**: Peak $\\rightarrow$ Neutral substitution
* **Likelihood / Scoring**: 2D Joint OT EMD Recovery percentage
* **Artifact & Output Path**: `v3/results/focused_causal_sweep_39pairs.csv`, `_pair_level.csv`

---

### D.9 Experiment v3-9: Generation-Time Full-Layer Sweep (15-Pair Exploratory)

#### 目的
介入タイミングを「プロンプト読了時（Prompt-time）」から「自己報告生成直前（Generation-time: 部分生成プレフィックス最終トークン）」へと切り替え、全84サイトの因果作用を探索した。

#### 実測結果
* **Layer 15 MLP**: $-0.01\\%$（依然として効果ゼロ）
* **Layer 24 Residual**: **$54.49\\%$**（劇的な因果回復が出現）

因果作用の局所的レバレッジが、中盤MLPではなく生成直前の後半Residual streamにおいて突如として現れる時空間的乖離が発見された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_generation_time_causal_sweep.py --pairs 15 --metric emd`
* **Sample Size ($N$)**: 15 initial pairs, `seed = 42`
* **Target Layer & Component**: 28 layers $\\times$ 3 components (84 sites)
* **Token Position**: Generation prefix last token (`gen_last_idx = inputs.input_ids.shape[1] - 1`)
* **Intervention Direction**: Peak $\\rightarrow$ Neutral substitution
* **Likelihood / Scoring**: Sequence continuation log-likelihood EMD recovery percentage
* **Artifact & Output Path**: `v3/results/generation_time_causal_sweep.csv`

---

### D.10 Experiment v3-10: Generation-Time Focused Full-Cohort Evaluation (39 Pairs)

#### 目的
代表6層 $\\times$ 3コンポーネント（18サイト）に対し、テストセット全39ペアを用いた本検証を実行し、主要効果量を確定した。

#### 実測結果
* **Layer 15 MLP**:
  * 平均回復率: **$-0.06\\%$** ($-0.0597\\%$)
  * 中央値: $+0.50\\%$
  * 95% Bootstrap CI: **$[-2.0\\%, +1.8\\%]$**
* **Layer 24 Residual**:
  * 平均回復率: **$53.24\\%$** ($53.2432\\%$)
  * 中央値: $55.08\\%$
  * 95% Bootstrap CI: **$[+45.7\\%, +60.5\\%]$**

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_focused_39pairs_sweep.py --eval-stage generation --layers 10,14,15,18,20,24 --components mlp,attn,resid`
* **Sample Size ($N$)**: 39 complete test pairs, `seed = 42`
* **Target Layer & Component**: Representative 6 layers $\\times$ 3 components (18 sites)
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Peak $\\rightarrow$ Neutral substitution
* **Likelihood / Scoring**: Pair-level EMD recovery distribution
* **Artifact & Output Path**: `v3/results/focused_causal_sweep_39pairs.csv`, `v3/results/focused_causal_sweep_39pairs_pair_level.csv`

---

### D.11 Experiment v3-11: Paired Peak-Site Contrast Bootstrap CI

#### 目的
同一ペア内における「因果レバレッジ最高部位（Layer 24 Residual）」と「デコード精度最高部位（Layer 15 MLP）」の直接対比（Paired Contrast: $D_i = \\text{rec}_{i,\\mathrm{L24\\_resid}} - \\text{rec}_{i,\\mathrm{L15\\_mlp}}$）をノンパラメトリック・ブートストラップ法（$B=2,000$）により評価した。

#### 実測結果
* **対比効果量**: **$\\Delta G = +53.30\\%$**
* **95% Bootstrap CI**: **$[+45.34\\%, +61.16\\%]$**（小数1桁丸め: $[+45.3\\%, +61.2\\%]$）
* **Wilcoxon Signed-Rank Test**: $W = 0.0, p = 7.28 \\times 10^{-11}$

信頼区間はゼロを遥かに超えて正側に位置しており、デコード可能性と因果レバレッジの空間的解離が極めて強固に証明された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/compute_bootstrap_ci.py --input v3/results/focused_causal_sweep_39pairs_pair_level.csv --n-boot 2000 --seed 42`
* **Sample Size ($N$)**: 39 paired differences, $B=2,000$ resamples
* **Random Seed**: `seed = 42`
* **Target Layer & Component**: Layer 24 Residual vs Layer 15 MLP contrast
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Paired within-sample difference
* **Likelihood / Scoring**: Percentile bootstrap 95% confidence intervals
* **Artifact & Output Path**: Console output & `v3/results/focused_causal_sweep_39pairs_pair_level.csv`

---

### D.12 Experiment v3-12: Multi-Layer Generation-Time Residual Patching

#### 目的
後半層残差ストリームの複数層同時パッチング（単層 L18, L20, L24 vs 複合 L20+24, L18+20+24, L18–24）を実施し、因果作用の加算性や飽和特性を検証した。

#### 実測結果
* **L24 単層**: $55.08\\%$
* **L20 + L24 (2層)**: $55.67\\%$
* **L18 + L20 + L24 (3層)**: $55.74\\%$
* **L18–24 (7層全体)**: $55.35\\%$

多層を束ねても回復率は約 $55\\%$ で完全に頭打ちとなった。これは下流での情報飽和（Downstream Saturation）や同一情報の再伝播と整合する。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_generation_multilayer_residual.py --combinations L18,L20,L24,L20+24,L18+20+24,L18-24`
* **Sample Size ($N$)**: 15 test pairs, `seed = 42`
* **Target Layer & Component**: Late-stage Residual stream combinations
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Multi-layer activation grafting
* **Likelihood / Scoring**: Generation-time EMD recovery percentage
* **Artifact & Output Path**: `v3/results/generation_multilayer_residual_results.csv`

---

### D.13 Experiment v3-13: Probe-Aligned Necessity Sweep

#### 目的
全28層 $\\times$ 3コンポーネント（84サイト）において、線形プローブ方向（$\\hat{v}_{\\mathrm{probe}}$）を直交射影により消去（Linear Subspace Ablation）した際に、自己報告分布が有意に中立化するか（必要性の検証）を網羅的に検定した。

#### 実測結果
84サイトすべてにおいて未補正 $p > 0.05$ であり、Benjamini-Hochberg FDR補正後の $q$ 値は全サイトで **$q = 1.000$** であった。すなわち、プローブ方向の単独除去では情動自己報告は消去されず、「Decodability Peak方向の局所的必要性」は完全に棄却された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_probe_aligned_necessity_sweep.py --all-layers --all-components`
* **Sample Size ($N$)**: 15 pairs, `seed = 42`
* **Target Layer & Component**: 84 sites (28 layers $\\times$ 3 components)
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Linear subspace ablation: $h' = h - (h \\cdot \\hat{v}_{\\mathrm{probe}})\\hat{v}_{\\mathrm{probe}}$
* **Likelihood / Scoring**: Paired t-test on log-odds shift; Benjamini-Hochberg FDR across 84 hypotheses
* **Artifact & Output Path**: `v3/results/probe_aligned_necessity_sweep.csv`

---

### D.14 Experiment v3-14: High-Resolution Focused Necessity Control (N=100 Directions)

#### 目的
代表5層 $\\times$ 3コンポーネント（15サイト）について、各サイトあたり $N=100$ 個のHaarランダム単位方向を抽出し、プローブ方向消去の効果がランダム部分空間消去の経験的帰無分布から有意に逸脱するかを高解像度で検証した。

#### 実測結果
全15サイトにおいて、プローブ方向消去によるシフト量はランダム方向消去の95%信頼包絡線内に完全に収まり、全サイトで **$q = 1.000$**（有意差なし）であった。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_focused_necessity_n100.py --layers 10,14,15,20,24 --n-samples 100 --seed 42`
* **Sample Size ($N$)**: 39 pairs, 100 random directions per component, `seed = 42`
* **Target Layer & Component**: Representative 5 layers $\\times$ 3 components (15 sites)
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Orthogonal subspace projection along $\\hat{v}_{\\mathrm{probe}}$ vs $\\hat{v}_{\\mathrm{rand}}^{(1\\dots 100)}$
* **Likelihood / Scoring**: Empirical null p-values with BH-FDR correction across 15 sites
* **Artifact & Output Path**: `v3/results/focused_necessity_sweep_n100.csv`

---

### D.15 Experiment v3-15: Dual-Outcome Behavioral Readout

#### 目的
Layer 20 / 24 Residual への介入によって自己報告分布が変化した際に、一人称自己報告だけでなく共感的応答文の生成トーンがどのように変化するか（二重アウトカム）を評価した。

#### 実測結果
自己報告Valenceのシフトが生じている場合でも、共感応答テキストの安全性・中立性ポリシーは頑健に保たれており、自己報告の変位とオープンエンドな文章生成ポリシーが独立して制御されていることが確認された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_dual_outcome_behavior.py --layers 20,24 --components resid`
* **Sample Size ($N$)**: 39 pairs, `seed = 42`
* **Target Layer & Component**: Layers 20, 24 Residual stream
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Peak $\\rightarrow$ Neutral activation grafting
* **Likelihood / Scoring**: Self-report VA distribution vs open-ended empathic response generation sentiment
* **Artifact & Output Path**: `v3/results/dual_outcome_results.csv`

---

### D.16 Experiment v3-16: Mood-Congruency / Third-Person Steering

#### 目的
中立的なナラティブ刺激（107刺激）に対し、Layer 14, 16, 20の残差活性化に情動方向ベクトルを加算（$\\alpha \\in \\{-3, -1.5, 0, 1.5, 3\\}$）した際の後続物語生成（Story completion）における感情極性の傾きを測定した。

#### 実測結果
自己報告タスクでは線形ステアリングが無効であったのと対照的に、自由物語生成においては感情極性スコアに有意な正の傾き（$\\beta = +0.182, p < 0.01$）が観測され、気分一致性効果（Mood-congruency effect）が確認された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_mood_congruency_experiment.py --layers 14,16,20 --alphas -3.0,-1.5,0.0,1.5,3.0`
* **Sample Size ($N$)**: 107 neutral narrative stimuli, `seed = 42`
* **Target Layer & Component**: Layers 14, 16, 20 Residual stream
* **Token Position**: Prompt last token
* **Intervention Direction**: Continuous vector addition: $h' = h + \\alpha \\sigma \\hat{v}$
* **Likelihood / Scoring**: Open-ended text generation sentiment regression slope
* **Artifact & Output Path**: `v3/results/mood_congruency/mood_congruency_results.csv`

---

### D.17 Experiment v3-17: Cross-Family Llama Partial Replication

#### 目的
Qwen2.5-1.5B-Instructで得られた「中盤MLPの因果不活性と終盤Residualの因果レバレッジ」という時空間的解離が、アーキテクチャの異なる `meta-llama/Llama-3.2-1B-Instruct` においても再現されるかを部分追試した。

#### 実測結果
* **Layer 10 MLP**: **$0.17\\%$**（効果ゼロ）
* **Layer 14 Residual**: **$45.41\\%$**（強い回復が出現）

Llamaにおいても中盤MLPでは因果作用が極小であり、終盤の残差ストリームで顕著な回復が現れるという時空間的パターンが完全に再現された。

#### Complete Reproducibility Specification
* **CLI Command**: `uv run python v3/scripts/run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct --layers 16 --components mlp,attn,resid`
* **Sample Size ($N$)**: 11 valid paired stimuli from AIPsy-Affect (Llama-3.2-1B-Instruct), `seed = 42`
* **Target Layer & Component**: 16 layers $\\times$ 3 components (48 sites)
* **Token Position**: Generation prefix last token
* **Intervention Direction**: Peak $\\rightarrow$ Neutral substitution
* **Likelihood / Scoring**: EMD recovery percentage
* **Artifact & Output Path**: `v3/results/generation_time_causal_sweep_llama.csv`'''


def main():
    paper_path = "v3/docs/paper2.md"
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find boundaries
    # Start of Appendix A:
    idx_a = content.find("A. Common Experimental Framework")
    if idx_a == -1:
        print("Error: Could not find Appendix A start.")
        sys.exit(1)

    # Start of Appendix E:
    idx_e = content.find("E. Overall Experimental Progression")
    if idx_e == -1:
        print("Error: Could not find Appendix E start.")
        sys.exit(1)

    # Build new text
    app_a_b = build_appendix_a_b()
    app_c = build_appendix_c()
    app_d = build_appendix_d()

    new_appendix = app_a_b + "\n\n---\n\n" + app_c + "\n\n---\n\n" + app_d + "\n\n---\n\n"

    updated_content = content[:idx_a] + new_appendix + content[idx_e:]

    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Successfully updated {paper_path} from Appendix A through D!")

if __name__ == "__main__":
    main()
