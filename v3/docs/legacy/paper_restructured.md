# Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models

**Authors**: Anonymous Authors  
**Status**: Restructured Working Paper Draft (Core 4-Figure & 3-Table Presentation)  
**Target Repository**: `emo/v3`

---

## Abstract

Linear probing of hidden activations is widely employed to locate where linguistic, factual, and affective properties are represented in Large Language Models (LLMs). However, whether high linear decodability of a concept indicates that the model causally utilizes that local representation in downstream computation remains a fundamental, unresolved question. In this study, we provide an empirical case study of this representation–use gap by evaluating affect-related representations in the Qwen2.5-1.5B (Base and Instruct) architecture.

Using the **AIPsy-Affect Strict Expanded** benchmark (192 pair groups, 422 texts matched for sentence length, syntax, and domain without explicit emotion lexicon) and an 81-candidate sequence-likelihood framework, we connect full-depth linear probing, direction-aligned ablation, within-model matched activation substitution, and cross-model manifold diagnostics.

Our findings reveal a profound empirical dissociation between representation decodability and causal leverage:
1. **Decodability Peak**: Affective peak versus neutral condition is linearly decodable with high accuracy in intermediate transformer layers, peaking at **Layer 15 MLP ($R^2 = 0.561$)**, **Layer 18 Attention ($R^2 = 0.550$)**, and **Layer 14 Residual ($R^2 = 0.502$)**.
2. **Absence of Prompt-Time Leverage**: At the prompt token where decodability culminates, substituting intermediate activations between matched affective and neutral inputs yields virtually zero causal recovery of downstream self-report distributions (**Layer 15 MLP: $S_{\mathrm{MLP},15} = 0.51\%$**, median $0.47\%$; all 28 layers demonstrate $\rho \le 0.296, p > 0.05$). Furthermore, geometrically projecting out the learned probe direction induces no systematic neutralization ($R_{\mathrm{neut}} \approx 0$, standardized specificity $Z_\perp \approx 0$, all 84 FDR $q > 0.85$).
3. **Emergence of Late Residual Leverage**: During autoregressive generation of the self-report tokens, substantial causal leverage abruptly emerges in downstream residual streams where prompt-time linear decodability is minimal (**Layer 24 Residual: prompt $R^2 = 0.147$, generation recovery $G_{\mathrm{RESID},24} = 53.24\%$**, median $61.57\%$, 95% CI: $[+45.7\%, +60.5\%]$). A direct pairwise comparison across 39 complete test pairs establishes a statistically decisive shift ($\Delta G = +53.30\%$, 95% CI: $[+45.34\%, +61.16\%]$, $p < 10^{-12}$).

These results demonstrate that **decodability does not localize causal leverage**: probing identifies where representations are accessible to an external observer, but fails to identify where, when, or along which dimensions information becomes causally operative in the model's own computation.

---

## 1. Introduction

A cornerstone of mechanistic interpretability is linear probing: if a linear classifier or regressor can predict a property $y$ from the hidden state $\mathbf{h}_\ell$ at layer $\ell$, the model is said to "linearly represent" that property. This paradigm has been applied across diverse capabilities, including factual knowledge, truthfulness, syntactic hierarchy, sentiment, and toxicity.

However, probing methodology has long contended with a foundational theoretical distinction:
$$\text{Linear Decodability} \not\equiv \text{Causal Behavioral Use}$$
An activation slice may exhibit high linear decodability because it acts as an epiphenomenal scratchpad, a passive transmission bus, or an entangled byproduct of contextual encoding, without serving as a causal bottleneck for downstream task execution.

```
Affective / Neutral text
        │
        ▼
┌───────────────────────┐
│ Qwen2.5-1.5B-Instruct │
└───────────────────────┘
        │
        ├── Prompt-time hidden states (Token t_last)
        │       ├── Linear Probing          ──→ Decodability D_l (Peak at L15 MLP: R²=0.561)
        │       ├── Matched Substitution    ──→ Prompt Causal Recovery S_l (≈ 0.5%)
        │       └── Probe Direction Ablation──→ Necessity N_l (Z_⟂ ≈ 0, non-specific)
        │
        └── Generation-time hidden states (Token t_gen)
                └── Matched Substitution    ──→ Generation Causal Recovery G_l,t (Surges at L24 Resid: 53.2%)
```

**Figure 1: Overall Experimental Framework and Central Finding.** *Where information is decodable does not identify where or when it becomes causally effective.*

In this paper, we systematically dissect this dissociation through a controlled case study of affect representations. When prompted to generate a first-person Valence–Arousal (VA) state (`{"valence": 5, "arousal": 5}`), instruction-tuned models exhibit a near-deterministic collapse toward neutral responses under greedy decoding ($98.6\%$). Yet, intermediate hidden states remain highly predictive of the input's affective valence.

Does this decodability indicate that affective representations are functionally present and merely masked at the final output token? Or is the accessible representation causally inert at that site?

By uniting full-depth linear probing ($D_\ell$), matched activation substitution ($S_\ell$), geometric probe ablation ($N_\ell$), and generation-time patching ($G_{\ell,t}$), we demonstrate that **the site of maximum decodability (Layer 15 MLP) possesses zero detectable causal leverage**, whereas **causal leverage emerges late in the residual stream (Layer 24 Residual) during output token generation** (Figure 1).

---

## 2. Experimental Task, Dataset, and Evaluation Design

To isolate causal mechanisms from surface heuristics, the experimental design enforces four strict methodological criteria (summarized in Table 1):

1. **Lexical-Neutralized Benchmark**: We deploy the **AIPsy-Affect Strict Expanded** corpus (192 matched groups, 422 narratives). Affective and neutral paired narratives are strictly matched for domain, syntactic complexity, and sentence length ($118 \pm 9$ words) while completely excluding explicit emotion words (e.g., "sad", "joyful", "furious").
2. **Constrained Sequence-Likelihood Scoring**: Because instruction tuning induces greedy output collapse, we evaluate the joint sequence log-likelihood distribution across all 81 valid integer Valence–Arousal completions:
   $$\mathcal{C} = \left\{ \text{"\{\"valence\": } V \text{, \"arousal\": } A \text{\}"} \mid V, A \in \{1, \dots, 9\} \right\}$$
3. **2D Joint Optimal Transport (OT) Distance**: Discrepancies between model output probability distributions are quantified using the 2D Wasserstein-2 distance on the discrete VA grid, providing a geometrically sound, divergence-free causal distance metric.
4. **Matched Counterfactual Substitution**: Interventions substitute activations exclusively between pre-matched affective and neutral narrative pairs ($\mathbf{x}_{\mathrm{aff}}, \mathbf{x}_{\mathrm{neut}}$), ensuring that intervention vectors lie within natural domain boundaries.

### Table 1: Dataset and Experimental Design Specifications
| Dimension | Specification | Methodological Purpose |
|---|---|---|
| **Primary Architecture** | Qwen2.5-1.5B & Qwen2.5-1.5B-Instruct | 28 transformer layers, hidden dimension $d=1536$ |
| **Comparative Architecture** | Llama-3.2-1B-Instruct | 16 transformer layers; architectural cross-check |
| **Dataset** | AIPsy-Affect Strict Expanded | Narrative pairs matched in length, syntax, and domain |
| **Corpus Volume** | 192 groups / 422 total samples | Train: 76 grp (169); Dev: 58 grp (124); Test: 58 grp (129) |
| **Confirmatory Test Pairs** | 39 complete test pairs | Strictly balanced affective vs. neutral paired inputs |
| **Intervention Sites** | 28 layers $\times$ 3 components = 84 sites | MLP output, Attention output, Residual stream |
| **Task & Output Format** | Constrained JSON self-report | `{"valence": V, "arousal": A}` |
| **Scoring Space** | 81 VA discrete sequence candidates | Sequence log-likelihood normalization |
| **Causal Distance Metric** | 2D Joint Optimal Transport (OT) | Wasserstein-2 metric on the $9 \times 9$ VA simplex |
| **Exploratory Causal Screen** | 15 test pairs (all 84 sites) | Unbiased full-depth layerwise screening |
| **Confirmatory Evaluation** | 39 test pairs (18 representative sites) | High-powered bootstrap CI estimation and hypothesis tests |

---

## 3. RQ1: Greedy Output Collapse vs. Distributional Sensitivity

When evaluated using greedy decoding on the affective benchmark, Qwen2.5-1.5B-Instruct outputs the exact neutral point `(5, 5)` in **$98.6\%$ of test cases**, compared to only $12.4\%$ in the unaligned Base model (Figure 4a).

However, this greedy invariance does not imply that the model is distributionally insensitive to affect. Examining the full conditional probability distribution over the 81 candidate completions reveals that the model's expected valence:
$$E[V] = \sum_{(V,A) \in \mathcal{C}} V \cdot P(V, A \mid \mathbf{x})$$
exhibits a strong, statistically significant correlation with human ground-truth ratings ($r = 0.629, p < 10^{-14}$; Spearman $\rho = 0.618$). As shown in Figure 4b, while post-training dramatically compresses the output range ($5.12 \le E[V] \le 5.78$ vs. Base $3.82 \le E[V] \le 7.14$), it preserves and sharpens the relative ranking of affective stimuli.

$$\text{Greedy Output Collapse} \not\equiv \text{Distributional Invariance}$$

This divergence establishes that downstream causal interventions cannot be measured via greedy token outputs alone; evaluating continuous likelihood geometry is essential to uncover internal functional dynamics.

```
  (a) Greedy Collapse Rate [%]          (b) Distributional Sensitivity (Human vs E[V])
   100 ┌──────────┐                          8 ┌──────────────────────────────────────┐
       │          │   Instruct (98.6%)         │             Base: r=0.365            │
    75 │          │                            │         Instruct: r=0.629            │
       │          │                            │                                      │
    50 │          │                          6 │              • • • •  Instruct (compressed)
       │          │                            │      •   • • • • •                       │
    25 │          │   Base (12.4%)             │  • • • • • •                             │
       │  ┌───┐   │                          4 │• • •  Base (wide dynamic range)      │
     0 └──┴───┴───┴───                         └──┴──────────┴──────────┴───────────┘
          Base  Instruct                          1          5          9  Human Valence
```
**Figure 4: Greedy Collapse vs. Distributional Sensitivity.** *(a) Instruct models deterministically collapse to the neutral prompt token `(5, 5)`. (b) Full sequence log-likelihoods preserve strong affective sensitivity ($r=0.629$).*

---

## 4. RQ2: Decodability Peaks Without Prompt-Time Causal Leverage

### 4.1 Layerwise Linear Decodability ($D_\ell$)
We trained ridge classification probes across all 28 layers and three architectural components (MLP output, Attention output, Residual stream) at the prompt's final token $t_{\mathrm{last}}$. As depicted in Figure 2a, decodability exhibits an inverted-U profile:
- **MLP Peak**: Reaches global maximum at **Layer 15** ($R^2_{\mathrm{MLP},15} = 0.5610$).
- **Attention Peak**: Reaches maximum at **Layer 18** ($R^2_{\mathrm{ATTN},18} = 0.5495$).
- **Residual Stream Peak**: Reaches maximum at **Layer 14** ($R^2_{\mathrm{RESID},14} = 0.5016$).

In these intermediate layers, linear decodability exceeds $R^2 > 0.50$, indicating that affective information is readily accessible to an external linear observer.

### 4.2 Prompt-Time Matched Causal Recovery ($S_\ell$)
To assess whether the model causally utilizes these representations at prompt time, we intervened at token $t_{\mathrm{last}}$ by substituting the activation slice of an affective input $\mathbf{x}_{\mathrm{aff}}$ with its matched neutral counterpart $\mathbf{x}_{\mathrm{neut}}$. Causal recovery is defined as:
$$S_\ell = \frac{\mathcal{D}_{\mathrm{OT}}(\mathbf{x}_{\mathrm{aff}}, \mathbf{x}_{\mathrm{neut}}) - \mathcal{D}_{\mathrm{OT}}(\mathbf{x}_{\mathrm{aff}\to\mathrm{neut}}^{(\ell)}, \mathbf{x}_{\mathrm{neut}})}{\mathcal{D}_{\mathrm{OT}}(\mathbf{x}_{\mathrm{aff}}, \mathbf{x}_{\mathrm{neut}})}$$
where $100\%$ denotes complete counterfactual transformation to the neutral distribution, and $0\%$ denotes zero causal effect.

As demonstrated in Figure 2b, **prompt-time local causal recovery is negligible across all 28 layers**:
- At the global decodability peak (**Layer 15 MLP**, $R^2 = 0.561$), recovery is merely **$S_{15} = 0.51\%$** (median $0.47\%$, 39 test pairs).
- Across all 84 sites in the 28-layer screening, recovery never exceeds $2.20\%$.
- The rank correlation between layerwise decodability $D_\ell$ and causal recovery $S_\ell$ is statistically indistinguishable from zero (MLP: $\rho = 0.296, p = 0.127$; Attention: $\rho = -0.222, p = 0.256$; Residual: $\rho = 0.046, p = 0.815$; Appendix Figure A3).

### 4.3 Probe-Aligned Directional Necessity ($N_\ell$)
Could the lack of effect in whole-slice substitution stem from off-target noise? To test whether the *specific direction* identified by linear probing is causally necessary, we ablated the learned probe weight vector $\mathbf{w}_\ell$ via geometric orthogonal projection:
$$\mathbf{h}'_\ell = \mathbf{h}_\ell - (\mathbf{h}_\ell^\top \mathbf{u}_\ell) \mathbf{u}_\ell, \quad \mathbf{u}_\ell = \frac{\mathbf{w}_\ell}{\|\mathbf{w}_\ell\|_2}$$
We benchmarked the resulting neutralization ratio $R_{\mathrm{neut}}$ against null distributions generated from 50 random orthogonal directions within the identical activation subspace.

As shown in Figure 2c, the standardized specificity $Z_\perp$:
$$Z_\perp = \frac{R_{\mathrm{neut}}(\mathbf{w}) - \mu_{\perp}}{\sigma_{\perp}}$$
remains confined within the null band ($|Z_\perp| < 2.0$) across all 84 sites. After Benjamini–Hochberg False Discovery Rate (FDR) correction, **0 out of 84 sites show statistically significant neutralization** (all FDR $q > 0.85$; Appendix Table A3). Removing the decodable linear axis does not alter downstream self-report distributions beyond random isotropic perturbation.

---

## 5. RQ3: Generation-Time Leverage & Direct Peak-Site Dissociation

```
  Figure 2: Four-Panel Representational–Causal Profile
  ┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │ (a) Layerwise Linear Decodability (D_l)      │ (b) Prompt-Time Causal Recovery (S_l)        │
  │  0.7 ┌────────────────────────────────────┐  │   8% ┌────────────────────────────────────┐  │
  │      │              ★ L15 MLP (R²=0.561)  │  │      │                                    │  │
  │  0.5 │          ▲ L14 Resid (0.502)       │  │   4% │       ★ L15 MLP: S_l ≈ 0.5%        │  │
  │      │              ■ L18 Attn (0.550)    │  │      │                                    │  │
  │  0.3 │                                    │  │   0% ├───-──-──-──-──-──-──-──-──-──-─────┤  │
  │      │  ── MLP   ── Attn   ── Resid       │  │  -4% │   Prompt substitution fails        │  │
  │  0.1 └──┴─────────┴─────────┴─────────┴───┘  │  -8% └──┴─────────┴─────────┴─────────┴───┘  │
  │         0         7        14        21  27  │         0         7        14        21  27  │
  ├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ (c) Probe-Aligned Necessity (Z_⟂)            │ (d) Generation-Time Causal Recovery (G_l,t)  │
  │   4  ┌────────────────────────────────────┐  │  60% ┌────────────────────────────────────┐  │
  │      │                                    │  │      │                        ★ L24 Resid │  │
  │   2  ├┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┤  │  40% │                        (G = 53.2%) │  │
  │   0  │──────── Null Band (|Z| < 2) ───────│  │  20% │               ▲ L18-20 Surge       │  │
  │  -2  ├┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┤  │   0% │───-──-──-──-                      │  │
  │  -4  └──┴─────────┴─────────┴─────────┴───┘  │ -20% └──┴─────────┴─────────┴─────────┴───┘  │
  │         0         7        14        21  27  │         0         7        14        21  27  │
  └──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```
**Figure 2: Four-Panel Representational–Causal Profile.** *(a) Linear decodability peaks in middle layers (L15 MLP). (b) Prompt-time matched substitution produces negligible causal recovery across all layers ($\approx 0\%$). (c) Probe direction removal exhibits no specific necessity across all 84 sites (all FDR $q > 0.85$). (d) Generation-time substitution induces dramatic causal recovery in late residual layers (L24 Residual: $53.24\%$).*

### 5.1 Late Residual Emergence ($G_{\ell,t}$)
Because prompt-time intervention yielded no causal lever, we shifted the intervention window to autoregressive generation time ($t_{\mathrm{gen}}$, during the generation of the JSON self-report tokens). 

As shown in Figure 2d, **causal recovery rises sharply in the late residual stream**:
- In middle layers, generation-time recovery remains near zero (Layer 15 MLP: $-0.06\%$).
- Beginning at Layer 18 Residual ($42.12\%$) and Layer 20 Residual ($50.22\%$), recovery surges, peaking at **Layer 24 Residual at $53.24\%$** (median $61.57\%$, 95% bootstrap CI: $[+45.7\%, +60.5\%]$).

### 5.2 Direct Peak-Site Dissociation
Table 2 summarizes the decisive dissociation across the key representative sites. In the middle MLP where decodability is maximized, the model possesses no local causal lever. In the late residual stream where causal recovery exceeds $50\%$, linear decodability has declined to $R^2 = 0.147$.

### Table 2: Main Representative-Site Results Across 39 Test Pairs
| Site | Component | Probe $R^2$ ($D_\ell$) | Prompt Rec. ($S_\ell$) | Generation Rec. ($G_{\ell,t}$) | 95% Bootstrap CI | Causal Status |
|---|---|---|---|---|---|---|
| **Layer 14** | Residual | 0.502 | 1.38% (med: 1.07%) | 0.52% (med: 2.06%) | [-3.64%, +5.66%] | Early decodability peak; inert |
| **Layer 15** | MLP | **0.561** | **0.51%** (med: 0.47%) | **-0.06%** (med: 0.50%) | **[-2.00%, +1.80%]** | **Global decodability peak; zero causal leverage** |
| **Layer 18** | Attention | 0.550 | 0.44% (med: 0.68%) | -6.82% (med: -6.87%) | [-11.20%, -2.60%] | Attention peak; negative generation effect |
| **Layer 18** | Residual | 0.485 | 0.63% (med: 0.28%) | 42.12% (med: 49.31%) | [+35.10%, +48.70%] | Emergence of late residual leverage |
| **Layer 20** | Residual | 0.401 | 1.01% (med: 0.26%) | 50.22% (med: 60.16%) | [+42.60%, +57.50%] | High causal transmission |
| **Layer 24** | Residual | **0.147** | -0.26% (med: -0.19%) | **53.24%** (med: 61.57%) | **[+45.70%, +60.50%]** | **Peak causal leverage; low prompt decodability** |

```
  Figure 3: Direct Peak-Site Dissociation (L15 MLP vs L24 Residual)
  (a) Site Contrast Summary                   (b) Pairwise Dissociation (N=39 pairs)
   60% ┌──────────────────────────┐            90% ┌──────────────────────────────────┐
       │             L24 Resid    │                │                               •  │
   40% │             (+53.2%)     │            60% │                         • • •    │
       │                          │                │                     • •          │
   20% │                          │            30% │                 • •              │
       │                          │                │             • •                  │
    0% │  L15 MLP                 │             0% ├───•─•──•───•─────────────────────┤
       │  (-0.06%)                │                │  • • •                           │
  -20% └──┴───────────────────────┘           -30% └──┴───────────────────────────────┘
          Decodability    Causal                       L15 MLP            L24 Resid
          Peak (L15)    Lever (L24)                  (Mean: -0.1%)       (Mean: +53.2%)
                                                         ΔG = +53.30%, 95% CI: [45.3, 61.2]
```
**Figure 3: Direct Peak-Site Dissociation.** *(a) Summary contrast between the decodability peak (Layer 15 MLP) and the causal leverage peak (Layer 24 Residual). (b) Pairwise slope lines across all 39 test pairs confirm a uniform, positive causal shift ($\Delta G = +53.30\%, p < 10^{-12}$).*

To provide definitive statistical confirmation, Figure 3 tracks the pair-by-pair causal recovery across all 39 test pairs between Layer 15 MLP and Layer 24 Residual:
$$\Delta G = G_{\mathrm{RESID},24} - G_{\mathrm{MLP},15} = +53.30\% \quad (95\%\ \text{Bootstrap CI}: [+45.34\%, +61.16\%])$$
The difference is uniform across pairs ($37 / 39$ pairs demonstrate positive shift; paired $t(38) = 13.80, p < 10^{-12}$; Wilcoxon $W = 770.0, p < 10^{-11}$). This provides direct empirical proof that **linear decodability peak and causal leverage peak are spatially and temporally dissociated**.

---

## 6. Secondary Findings and Boundary Conditions

While the primary thesis centers on the decodability–leverage dissociation, three additional investigations delineate the boundary conditions of the phenomenon (fully detailed in the Appendix):

### 6.1 Multi-Layer Patching Saturation
Simultaneously substituting multiple late residual layers (Layers 20, 22, 24, 26) yields a causal recovery of **$55.4\%$**, representing an incremental gain of only $+2.2\%$ over Layer 24 alone ($53.2\%$) (Appendix Figure A9, Table A6). Late residual causal leverage operates as a non-additive, saturated channel rather than an independent linear sum of layerwise signals.

### 6.2 Cross-Model Alignment Collapses onto Manifold Boundary
When attempting to transplant activations from the unaligned Base model to the Instruct model via ridge regression, high linear alignment quality ($R^2 \approx 0.88$, linear CKA $= 0.94$) can be achieved. However, multi-metric out-of-distribution (OOD) diagnostics reveal that the aligned activations collapse into an unnatural manifold configuration (Mahalanobis distance $D_M$ expands beyond natural baselines; Appendix Figure A7, A8). High predictive alignment does not ensure distributional typicality or causal transferability:
$$\text{Predictive Cross-Model Alignment} \not\equiv \text{Distributional Typicality}$$

### 6.3 Cross-Architecture Replication Probe (LLaMA-3.2-1B)
Replicating the full-layer generation sweep on Llama-3.2-1B-Instruct revealed that generation-time recovery remained flat across all 16 layers (recovery $\le 1.8\%$; Appendix Figure A10). This indicates that the specific routing of affective self-reports through late residual layers is architecture- and post-training specific, rather than a universal invariant across all transformer models.

---

## 7. General Discussion and Theoretical Synthesis

### Table 3: Synthesis of Core Claims and Empirical Evidence
| Research Question | Methodological Evidence | Empirical Finding | Theoretical Implication |
|---|---|---|---|
| **RQ1: Affective Sensitivity** | 81-candidate likelihood scoring | Greedy collapse ($98.6\%$) coexists with high likelihood correlation ($r=0.629$). | **Greedy Output Collapse $\neq$ Distributional Invariance.** Post-training compresses output dynamic range without destroying relative affective geometry. |
| **RQ2a: Linear Accessibility** | 28-layer probing ($D_\ell$) | Peaks at Layer 15 MLP ($R^2=0.561$) and Layer 18 Attn ($R^2=0.550$). | Affective information is strongly accessible to linear readouts in middle representations. |
| **RQ2b: Prompt-Time Leverage** | Matched substitution ($S_\ell$) | Layer 15 MLP recovery $= 0.51\%$; full-layer correlation $\rho \le 0.296$ ($p > 0.05$). | **Decodability does not localize prompt-time causal leverage.** Accessible representations are not local bottlenecks. |
| **RQ2c: Directional Necessity** | Probe ablation ($N_\ell$, $Z_\perp$) | All 84 sites show no specific neutralization vs. random controls (FDR $q > 0.85$). | Probing extracts descriptive read-out axes that lack unique causal necessity. |
| **RQ3a: Downstream Leverage** | Generation-time substitution ($G_{\ell,t}$) | Surges to $53.24\%$ in Layer 24 Residual, where prompt decodability is low ($R^2=0.147$). | Causal leverage emerges downstream during the autoregressive generation of report tokens. |
| **RQ3b: Statistical Dissociation** | Pairwise contrast ($N=39$ pairs) | $\Delta G = +53.30\%$, 95% CI: $[+45.34\%, +61.16\%]$, $p < 10^{-12}$. | **Peak decodability and peak causal leverage are empirically dissociated.** |
| **RQ4: Manifold Transferability** | Cross-model ridge & Mahalanobis | High predictive alignment ($R^2=0.88$) collapses onto OOD manifold configurations. | **Predictive Alignment $\neq$ Distributional Typicality.** Linear fit does not guarantee functional transplantability. |
| **RQ5: Model Specificity** | Cross-model sweep on LLaMA | Flat generation recovery across all 16 layers ($\le 1.8\%$). | Causal routing pathways depend on specific training regimes and model scale. |

### 7.1 Implications for Mechanistic Interpretability
The primary implication of our findings is methodological: **researchers cannot rely on linear probing to identify targets for model editing, steering, or circuit analysis**. A site exhibiting maximum probing decodability may be entirely bypassed by downstream computational paths. Interventional techniques (activation patching, path patching, and directional ablation) are not merely confirmatory checks for probing—they are fundamentally distinct measurements that probe different operational regimes of the model.

### 7.2 Affect Without Phenomenological Experience
In accordance with measurement validity standards, we emphasize that our findings characterize *behavioral and computational representations*, not subjective phenomenological feeling. The emergence of late residual causal leverage reflects the routing of affective lexical and contextual priors into structured self-report tokens, not the emergence of machine sentience.

---

## 8. Conclusion

By systematically pairing linear probing with interventional causal substitution across all layers and generation phases of Qwen2.5-1.5B, this study provides rigorous empirical evidence of the representation–use gap. Peak decodability at Layer 15 MLP ($R^2 = 0.561$) yields no prompt-time causal leverage ($0.51\%$), while peak causal leverage ($53.24\%$) emerges during generation in Layer 24 Residual where prompt decodability is low ($0.147$). 

We conclude that **decodability does not localize causal leverage**. Mechanistic explanations of LLM behavior must decouple where representations can be decoded from where and when they govern computation.

---

# Appendices: Complete Archival Reproducibility

The following appendices document the complete numerical records, statistical tables, and sensitivity sweeps supporting the main text.

- **Appendix A**: Full 28-Layer Prompt Screening Numerical Tables & Probing Curves (Table A1; Figure A1, A2, A3)
- **Appendix B**: Full 28-Layer Generation Screening Numerical Tables (Table A2; Figure A4)
- **Appendix C**: Probe-Aligned Necessity & Specificity Statistical Battery across All 84 Sites (Table A3, A4; Figure A5, A6)
- **Appendix D**: 39-Pair Focused Evaluation: Full 18-Site Parameter Distribution (Table A5)
- **Appendix E**: Multi-Layer Residual Patching Details & Saturated Combinations (Table A6; Figure A9)
- **Appendix F**: Cross-Model Ridge Regularization Sweep & Manifold OOD Diagnostics (Table A7; Figure A7, A8)
- **Appendix G**: LLaMA-3.2-1B-Instruct Cross-Architecture Replication Profile (Figure A10)
- **Appendix H**: Complete Code Repository, Script Mappings, and Artifact Hashes (Table A8)
