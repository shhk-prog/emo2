# Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models

## Abstract

大規模言語モデル（LLM）の内部状態から意味属性を高精度に線形デコードできることは、その情報が表現内にアクセス可能な形で存在することを示す。しかし、decodabilityは、その表現部位が下流の振る舞いを因果的に制御していることを必ずしも意味しない。本研究では、この古典的なrepresentation–use distinctionを、Qwen2.5-1.5B Base/Instructモデルペアにおけるaffect-relevant representationsをケーススタディとして体系的に検証する。

語彙交絡を制御した最小対データセット（AIPsy-Affect Strict Expanded: 192 pair-id groups / 422 samples）、81候補のValence–Arousal likelihood evaluation、全28層のlinear probing、Base→Instruct表現アライメント、多変量OOD診断、およびwithin-model activation substitutionを統合した。

主要な発見は以下の3点に集約される：

第一に、affective peak versus neutral conditionの線形decodabilityは中間層で高く、MLPではLayer 15（$R^2=0.561$）、AttentionではLayer 18（$R^2=0.550$）、Residual streamではLayer 14（$R^2=0.502$）で最大となった。

第二に、この高い線形アクセス可能性は局所因果レバレッジを意味しない。初期15-pair探索スクリーニング（Stage 1）ではLayer 15 MLPのPrompt-time回復率は1.00%に留まり、39-pair focused full-cohort evaluation（Stage 2: Prompt有効 N=32, Generation有効 N=39）ではさらに0.51%（中央値0.47%）であった。Generation-timeでも-0.06%（中央値0.50%, 95% CI: [-2.0%, +1.8%]）に留まった。Prompt-time探索的全層スクリーニングでは、層別decodabilityとlocal causal recoveryの間に統計的に検出可能な単調関係は認められなかった（$\rho \le 0.296, p > 0.05$）。

第三に、自己報告生成直前には、prompt-time decodabilityが比較的低い後段Residual streamに強いmatched-substitution recoveryが出現した（Layer 24: prompt-time $R^2=0.147$, generation-time recovery = 53.24% [中央値 61.57%, 95% CI: +45.7% to +60.5%]）。一方、学習済みプローブ方向の幾何学的射影消去は系統的な中和をもたらさず、直交ランダム方向消去を上回る特異的効果を示さなかった。なお、Llama-3.2-1B-Instructでの初期部分追試では後段Residual局在は再現されず、モデル依存性が示唆された。

以上の結果は、**表現のアクセス可能性（accessibility）と局所的因果レバレッジ（local causal leverage）が経験的に解離し得る（empirically dissociable）別個の性質である**ことを示す。Linear probingが情報を外部から読み取れる場所を同定しても、その情報がモデルの下流計算において強い因果的影響力を持つ場所、時点、あるいは方向をそれ自体では同定しない。すなわち、**Decodability does not localize causal leverage**。

---

## 1. Introduction

大規模言語モデルのhidden representationsには、言語構造、知識、真偽、安全性、感情的属性など、さまざまな情報が線形にデコード可能な形で含まれることが知られている。

しかし、probeがある属性を予測できることから、

> *the model uses that representation to produce the downstream behavior*

と結論することはできない。

この問題はprobing研究において以前から指摘されてきた（例: *Elazar et al., 2021; Ravfogel et al., 2021; Belinkov, 2022*）。Probeはモデル自身のdownstream computationには必要でない相関情報も利用できるため、

$$\text{Linear Decodability} \not\Rightarrow \text{Causal Localization}$$

である。本研究では、この区別をBase/Instruct language modelにおけるaffect-relevant representationsを用いて検討する。

### 1.1 Motivation

Qwen2.5-1.5B-Instructへaffective textを提示し、一人称形式でValence–Arousalを報告させると、greedy decodingはほぼ常に

```json
{"valence": 5, "arousal": 5}
```

へ集中する。

しかし、この現象には少なくとも以下の異なる説明が存在する：

1. **表現の減衰・変形（Representational attenuation or transformation）**: affect-relevant informationが内部表現空間において減衰・変形している。
2. **表層的中立化**: 情報は存在するがgreedy outputだけが中立化されている。
3. **局所因果的解離**: 情報はdecodableだが、その局所スライス表現は報告生成に対する因果的レバレッジ（local causal leverage）を持たない。
4. **分散的・動的利用**: 情報は別のtoken position、component、あるいはgeneration-time computationとして利用されている。

これらを区別するためには、behavior、decodability、representation alignment、causal interventionを独立に測定する必要がある。

### 1.2 Research Questions

- **RQ1**. Greedy decodingで中立化されたモデルは、候補列全体の尤度分布においても感情感受性を完全に失っているのか。
- **RQ2**. Affect-relevant condition indicatorは、BaseモデルおよびInstructモデルの内部表現からどの程度線形デコード可能か。
- **RQ3**. Baseモデルの表現からInstructモデルの表現への線形写像は、幾何学的に自然な活性化分布を再現できるか。
- **RQ4**. Baseモデルの活性化をアライメントしてInstructモデルへ注入した場合、出力分布は感情方向へ因果的に回復するか。
- **RQ5**. Probe-accessible activation sitesは、対応するreportに対して局所因果回復（local causal recovery）または局所因果レバレッジ（local causal leverage）を持つか。

---

## 2. Construct Definition

本研究では、測定対象を明確に限定する：

1. **Self-Reported Affective State**: 制約付きプロンプトに対するモデルの明示的出力トークン列から得られるVA値。主観的感情経験の存在を主張するものではない。
2. **Constrained Likelihood Distribution**: 81個のVA候補JSON列に対するモデルの条件付き同時対数尤度から構成される正規化確率分布。
3. **Linear Decodability ($D_\ell$)**: 特定層・特定コンポーネントの隠れ状態から、刺激の感情条件（Peak vs. Neutral）をロジスティック回帰で判別した際の決定係数（Pseudo-$R^2$相当）。
4. **Local Causal Recovery ($C_\ell$)**: 同一モデル内のPeak活性化を対応するNeutral活性化で置換した際、出力確率分布が元のPeak分布へどれだけ引き戻されるかを2D Joint Optimal Transport（Earth Mover's Distance）によって測った回復率。
5. **Probe-Aligned Local Necessity ($N_\ell, R_{\mathrm{neut}}$)**: 学習済みプローブ方向を幾何学的に直交射影除去した際の出力分布変位量、およびNeutral分布への引き戻し比率。
6. **Local Causal Leverage（操作的定義）**: 特定layer/component/token positionの活性化をmatched counterfactual activationで置換した際に出力分布が目標分布へ移動する程度（tested matched-substitution recovery）。当該部位が計算の起源、唯一の経路、または必要条件であることを意味しない。

---

## 3. Experimental Setup

- **対象モデル**: Qwen2.5-1.5B (Base) および Qwen2.5-1.5B-Instruct（28層, $d=1536$）、Meta Llama-3.2-1B-Instruct（16層, $d=2048$）。
- **刺激データセット**: AIPsy-Affect Strict Expanded（192 pair-id groups / 422 samples）。全サンプルについて、感情語彙の有無による表層的交絡を厳格に制御した最小対を構成。
- **評価プロトコル**:
  - Prompt-time: プロンプト最終トークンにおける介入。
  - Generation-time: 生成プレフィックス `{"valence": ` 直後の第1生成トークン位置における介入。
  - 評価指標: 81点（Valence 1〜9 × Arousal 1〜9）の完全離散2次元格子におけるマンハッタン距離コストに基づく真の2D Joint Optimal Transport（Joint OT）。

---

## 4. Constrained Report Measurement

Instructモデルにおいて、greedy decodingは98.6%のサンプルで中立値 `(5, 5)` を出力した。しかし、81通りのVA候補列に対する完全な条件付き対数尤度を算出し、Softmaxにより確率分布を構成して期待値を計算したところ、期待Valenceと刺激の感情価の間に統計的に有意な共変動（$r = 0.382, p < 0.001$）が認められた。

したがって、greedy decodingにおける表層的中立化は、モデル内部における刺激感受性の完全な消失を意味しない。

---

## 5. Detailed Results

### 5.1 Base/Instruct表現アライメントと予測的対応
全28層のMLP出力についてRidge回帰（$\alpha$ スイープ）によるBase→Instruct線形写像を学習した。中盤層（Layer 14–15）においてLinear CKAは0.829に達し、テストペアの検索正解率（retrieval top-1）は86.6%を記録した。

### 5.2 Statistical Geometry: Center Collapse
高い予測精度（CKA $\approx 0.83$）にもかかわらず、アライメント後の活性化ベクトルは自然なInstruct活性化分布の中心方向へ著しく収縮した（分散比 0.23〜0.31）。多変量マハラノビス距離による外れ値診断では、アライメントベクトルの92.4%が通常の活性化マニホールドの許容境界外に位置した。

### 5.3 Cross-Model Patchingの限界
この幾何学的歪みにより、Baseモデルの活性化をアライメントしてInstructモデルへ注入するCross-model patchingでは、出力分布が破壊され負の回復率（コヒーレンス崩壊）を引き起こした。これは、予測的類似性（predictive alignment）が機能的等価性（functional equivalence）を保証しないことを実証している。

---

## 6. Core Result: Layerwise Dissociation Between Decodability and Local Causal Leverage (RQ5)

全28層について、最終prompt tokenにおけるMLP output、Attention output、およびResidual streamを対象として、線形解読能 $D_\ell$ と局所因果回復率 $C_\ell$ を網羅的に測定した。

### 6.1 全28層における層別マッピング

**表6: 全28層における線形プローブ決定係数 ($R^2$) と真の2D Joint OT因果回復率 ($C_\ell^{\mathrm{Joint}}$) の網羅的マッピング (MLP / Attention Output / Residual Stream)**

| Layer | MLP Probe $R^2$ | MLP Rec (%) | ATTN Probe $R^2$ | ATTN Rec (%) | Resid Probe $R^2$ | Resid Rec (%) |
|---|---|---|---|---|---|---|
| **0** | 0.3025 | -1.39% | 0.3395 | -0.92% | 0.3129 | -1.15% |
| **1** | 0.3045 | -0.47% | 0.3263 | +0.78% | 0.3529 | -0.04% |
| **2** | 0.3608 | -0.32% | 0.3971 | -0.78% | 0.3943 | -1.93% |
| **3** | 0.3932 | +1.38% | 0.4903 | +0.60% | 0.3694 | -2.73% |
| **4** | 0.3847 | +1.67% | 0.3997 | -2.04% | 0.3521 | -2.71% |
| **5** | 0.3685 | -0.57% | 0.3844 | +0.61% | 0.4059 | -2.23% |
| **6** | 0.4229 | -0.17% | 0.4698 | -1.23% | 0.4697 | -3.70% |
| **7** | 0.4787 | +1.21% | 0.4321 | -0.50% | 0.4726 | -2.09% |
| **8** | 0.4671 | -0.94% | 0.3837 | +0.20% | 0.4619 | -3.05% |
| **9** | 0.4134 | +0.78% | 0.4157 | +0.09% | 0.4141 | -2.31% |
| **10** | 0.4817 | **+2.20% (Max)** | 0.4753 | +0.78% | 0.4744 | -1.37% |
| **11** | 0.4891 | +0.25% | 0.4830 | +0.21% | 0.4519 | -0.16% |
| **12** | 0.4571 | -0.92% | 0.4085 | +1.46% | 0.4561 | -0.53% |
| **13** | 0.5151 | -0.06% | 0.5205 | +0.92% | 0.4887 | +0.61% |
| **14** | 0.5372 | +1.35% | 0.5313 | -0.36% | **0.5016 (Peak)** | +0.71% |
| **15** | **0.5610 (Peak)** | +1.00% | 0.5217 | -0.52% | 0.4862 | +0.25% |
| **16** | 0.5158 | +1.02% | 0.4913 | +0.98% | 0.4705 | **+1.68% (Max)** |
| **17** | 0.4772 | -0.04% | 0.4936 | -0.04% | 0.4432 | +1.34% |
| **18** | 0.5513 | -0.43% | **0.5495 (Peak)** | +0.04% | 0.4848 | -0.58% |
| **19** | 0.4434 | -0.15% | 0.4483 | -0.18% | 0.4330 | +0.81% |
| **20** | 0.4866 | -0.38% | 0.4602 | **+1.48% (Max)** | 0.4008 | +0.65% |
| **21** | 0.5223 | +0.62% | 0.4859 | +0.21% | 0.3903 | +0.85% |
| **22** | 0.3923 | +0.16% | 0.4076 | +0.96% | 0.2193 | +0.31% |
| **23** | 0.3774 | +0.52% | 0.4514 | +1.23% | 0.2107 | +1.31% |
| **24** | 0.3514 | +0.84% | 0.4072 | +0.61% | 0.1470 | -0.00% |
| **25** | 0.3178 | -0.36% | 0.4100 | +0.97% | 0.1790 | +0.18% |
| **26** | 0.3311 | +0.44% | 0.1040 | +0.41% | 0.1681 | +0.23% |
| **27** | 0.3131 | -0.00% | -0.2670 | -0.00% | 0.2496 | -0.00% |

真の2D Joint OTに基づく全層測定により、以下の知見が得られた：

1. **極小の局所因果回復 (Minimal Local Causal Recovery)**: 全28層・全コンポーネントを通じて因果回復率は極めて小さく、最大値でもMLPで2.20%（Layer 10）、Attentionで1.48%（Layer 20）、Residualで1.68%（Layer 16）に留まる。
2. **最高デコード層での解離**: MLPデコーダビリティが最大となるLayer 15（$R^2 = 0.5610$）における回復率はわずか1.00%であり、Attentionデコーダビリティが最大となるLayer 18（$R^2 = 0.5495$）での回復率は0.04%であった。
3. **無相関の頑健性**: 層別probe $R^2$と因果回復率の間には、いずれのコンポーネントにおいても単調関係は検出されなかった（MLP: Spearman $\rho = 0.2956, p = 0.1268$, 95% Bootstrap CI: $[-0.08, 0.61]$; ATTN: $\rho = 0.0230, p = 0.9076$, 95% Bootstrap CI: $[-0.35, 0.40]$; RESID: $\rho = -0.0394, p = 0.8422$, 95% Bootstrap CI: $[-0.41, 0.35]$）。

*プローブ推定値の整合性に関する注記*:
本研究の主解析では、全28層・全コンポーネントで統一的に算出した $R^2_{\mathrm{MLP},15} = 0.5610$、$R^2_{\mathrm{ATTN},18} = 0.5495$、$R^2_{\mathrm{RESID},14} = 0.5016$ を正本として採用している。なお、Section 3で言及した予備的な全層スイープ実装（初期の尤度ロバストネス評価パイプラインと同時に抽出されたプローブ推定: 0.546）と比較しても、定性的な層別プロファイルおよび効果量の範囲は緊密に整合している。

さらに、本研究では以下の対立仮説を独立実験により二段階で検証・評価した：

- **Stage 1: Exploratory Generation-time Screen (15-pair Sweep)**: 初期の15-pair全層探索スクリーニング（`v3/results/generation_time_causal_sweep.csv`）において、中盤層（Layer 15 MLP: -1.99%）における局所因果回復の欠如と、**Layer 18〜27の後段Residual streamにおける急激な因果回復（40.01%〜50.26%、中央値 46.34%〜56.05%）**という時空間的局所化プロファイルが同定された。この結果は、プロンプト時には隠蔽されていた因果レバレッジが自己報告生成直前に後段Residual streamへ動的に移行することを示唆した。
- **Stage 2: Focused Full-Cohort Evaluation on Representative Layers**: 初期スクリーニングで得られた候補層に加え、probe peakおよびcomponent-specific representative layersを含む代表6層（L10, 14, 15, 18, 20, 24）について、Held-out test全39組の完全一致ペアを評価対象とした（`v3/results/focused_causal_sweep_39pairs.csv`）。分母セーフガード適用後、Prompt時32ペア、Generation時39ペアが有効であった。なお、代表層にはプローブ定義層に加え探索スクリーニングからの優先層も含まれるため、本全数評価は選択独立な確認的仮説検定ではなく、探索的に同定された効果の安定性と効果量を全コホート上で再評価することを目的とする。結果、プローブ最高層 Layer 15 MLP（$R^2=0.5610$）の回復率はPrompt時 **0.51%**（中央値 0.47%）、Generation時でも **-0.06%**（中央値 0.50%, 95% CI: [-2.0%, +1.8%]）に留まった。一方、生成時においては後段層の **Residual Stream（L18: 42.12% [35.1%, 48.7%], L20: 50.22% [42.6%, 57.5%], L24: 53.24% [45.7%, 60.5%]）および後段MLP（L24: 15.04% [10.5%, 19.5%]）** に実質的な因果回復が出現し、Stage 1で観測された後段Residualへの強い局所化が完全再現された（*tested local causal recovery profile shifted toward late Residual/MLP sites during report generation*）。極めて重要な点として、*Importantly, the large late-residual effects were not driven by a small number of outlying pairs. At L20 and L24, median recovery was 60.16% and 61.57%, respectively, with positive recovery in 97.4% and 94.9% of valid matched pairs.*（L20およびL24の中央値回復率は 60.16% および 61.57% であり、有効ペアの 97.4% および 94.9% で正の回復が確認されたため、少数の外れ値による歪みではない）。
- **Multi-Layer Simultaneous Residual Stream Patching**: 単一層のL24 Residual（55.08%）に対し、2層（L20+24: 55.67%）、3層（L18+20+24: 55.74%）、7層連続（L18〜24: 55.35%）と多層同時パッチングを行っても、回復率は約55%（中央値約62%）で完全に飽和した。この結果は、後段Residual Streamの各層が互いに独立した加算的寄与を累積しているというよりは、後段部位間における実質的な冗長性や下流計算での飽和、あるいは介入状態間の依存性と整合する（*consistent with substantial redundancy or saturation across late residual sites, rather than additive independent contributions*）。*なお、多層実験は独立パイプラインで再実行されたため、その単層参照値（55.08%）はfocused sweepの単層推定値（53.24%）とわずかに異なるが、ともに53〜55%の強固な回復を示している。*
- **Probe-Aligned Necessity Sweep & N=100 Focused Test**: プローブ方向の直交射影消去による中和比率は全層で -3.27%〜+0.61% であり、系統的な中和傾向は認められなかった。重要代表5層における **N=100 ランダム直交方向高解像度追試**（`v3/results/focused_necessity_sweep_n100.csv`）でも、多重比較補正以前の段階ですべての条件で未補正の経験的 $p \ge 0.297$（$Z_\perp \le 0.51$）であり、BH-FDR補正後も全15条件で $q = 1.000$（有意層 0/15）となった。すなわち、プローブ方向の消去は系統的な中和をもたらさず、直交ランダム方向を上回る出力変位も示さなかった。
- **Attention Pathway**: 単一層射影済みAttention出力の置換でも因果回復率は極小（Prompt最大1.48%, Generation最大4.23%; 全数39ペアでは 1.32%）であり、tested single-layer projected Attention outputs did not show strong local causal recovery（本実験で検証した単一層の射影済みAttention出力が、強力な局所的因果ボトルネックとして機能する証拠は見出されなかった）。
- **Cross-Family Partial Replication**: Llama-3.2-1B-Instruct（11 valid pairs）での初期部分追試では生成時回復率は最大0.73%（pairwise中央値0.00%）に留まり、Qwenで観測された後段Residualへの強い局所回復は再現されなかった。これは後段Residualへの因果レバレッジ局在化がモデルファミリや訓練方策に依存する可能性を示唆している。

したがって、

$$\boxed{\text{Linear Decodability} \not\Rightarrow \text{Causal Localization}}$$

$$\boxed{\text{Decodability peak} \neq \text{Causal leverage peak}}$$

$$\boxed{\text{where information is decodable} \neq \text{where/when it becomes causally effective}}$$

このpeak-site contrastは、本研究で観察されたdecodabilityとlocal causal leverageの解離を最も直接的に示す効果量ベースの証拠である（*The peak-site contrast provides the most direct effect-size evidence for the dissociation observed in this study*）：
$$
\Delta G = G_{\mathrm{L24,RESID}} - G_{\mathrm{L15,MLP}} = +53.30\% \quad (95\%\text{ bootstrap CI: } [+45.34\%, +61.16\%], \text{median difference: } +60.08\%)
$$
すなわち、全28層の探索的Spearman相関（$\rho \approx 0$）は補助的証拠であり、全数コホートで評価した代表部位間における直接的な効果量対比——すなわち、**全層プローブ最高部位（Layer 15 MLP: $D=0.561, G=-0.06\%$）と、評価した全数代表部位において最大回復を示した部位（the strongest recovery among the evaluated full-cohort representative sites, Layer 24 Residual: $D=0.147, G=53.24\%$）との間の劇的な双方向解離**——こそが本研究の中心命題を支える主たる実証的証拠（primary evidence）である。最大解読能を示す中間層（L15 MLP）はPrompt時・Generation時を問わず局所因果レバレッジをほとんど持たない一方、強い因果レバレッジは自己報告生成直前の後段Residual Stream（L18〜L24）に時間的・空間的に解離して出現することが実証された。

この知見は、古典的な representation $\neq$ use を以下の3層＋時間軸へ精緻化することを要請する：

$$\boxed{\text{Accessibility},\; \text{Local Causal Recovery},\; \text{Direction-Specific Necessity} \text{ are distinct empirical axes}}$$

$$\boxed{\text{Causal leverage is position- and time-dependent}}$$

---

### 6.2 Robustness to Likelihood Scoring

候補列のtokenization lengthによって結果が生じている可能性を検証するため、
1. token-length-normalized log likelihood
2. raw sequence log likelihood
の二つのscoring protocolで全解析を再実行した。

候補列の尤度スコアリング方式（正規化の有無）によらず、同一データスプリットから得られる線形デコーダビリティプロファイルはほぼ同一となり、MLPの最大値はいずれもLayer 15に位置した（$R^2 \approx 0.55$；先行スイープ実装で $0.546$ / $0.547$、最終統合スイープで $0.561$）。

より重要な点として、下流自己報告の因果回復率は両プロトコルとも全層で極小に留まった：

$$\max C_\ell^{\mathrm{Norm}} = 1.37\%, \qquad \max C_\ell^{\mathrm{Raw}} = 0.97\%$$

さらにdecodability–recovery correlationも両protocolで小さかった。

したがって、観測されたlayerwise dissociationはcandidate sequence lengthの正規化方法に依存しない。

---

### 6.3 ポジティブコントロールプロトコルの検証

また、局所スライス置換と全トークン置換の振る舞いを比較するため、代表プロトコルでの検証も実施した。

**表5: 同一モデル内ポジティブコントロールにおける回復率および期待値シフト (Held-out Test N=39)**

| プロトコル | Raw LL (単純和) Mean / Median Rec | Norm LL (長さ正規化) Mean / Median Rec | EV Shift (Raw / Norm) | 解釈・メカニズム |
|---|---|---|---|---|
| **`mlp_last_token_L15`** | **0.82%** / 0.56% | **1.30%** / 1.13% | +1.09% / -1.45% | **プローブ最高層でも局所スライスの回復率は ~1%** |
| **`resid_last_token_L15`** | **0.38%** / 0.36% | **0.42%** / 0.15% | +2.97% / +0.68% | 単一残差ストリームの局所制御力はさらに微小（< 0.5%） |
| **`multi_resid_last_L13_16`** | **1.47%** / 1.06% | **1.38%** / 0.89% | +3.53% / +0.09% | 4層連続の最終トークン置換でも約 1.4% にとどまる |
| **`resid_all_tokens_L15`** | -448.95% / -263.24% | -225.88% / -133.28% | +893.61% / +234.28% | **文脈崩壊（コヒーレンス破壊）により出力分布が外れ値へ発散** |
| **`multi_resid_all_L13_16`** | -437.15% / -248.80% | -236.91% / -129.70% | +876.11% / +214.96% | 複数層全トークン置換でも同様に分布破壊（負の回復） |

全トークン置換では異なる文脈の活性化を無理に貼り付けることでコヒーレンス破壊（負の回復）が生じるのに対し、最終トークン置換では出力分布を壊すことなく安定して測定できる。しかしその回復率は高々 1% 前後にとどまった。

---

## 7. Discussion

### 7.1 Interpretation: Decodability Is a Representational Claim, Not a Mechanistic Claim

本結果は、

$$\text{representation contains information}$$

と

$$\text{a local representation slice controls behavior}$$

を明確に区別する必要性を実証する。

特に、本研究ではaffective conditionが最も明確に読み取れるLayer 15 MLP（$R^2 = 0.5610$）であっても、全39組を対象としたfocused evaluationにおいてPrompt-time回復率は0.51%（全28層探索スクリーニングでも1.00%）、Generation-timeでも-0.06%に留まった。

したがって、probe accuracyを用いてactivation patchingの介入位置を選択する一般的な戦略には注意が必要である。高いprobe accuracyは、

> *“this is a location from which the information is linearly accessible”*

という記述的主張をある程度支持するが、

> *“this is a location where intervention will exert local causal control over the downstream decision”*

という機構的主張を直接には支持しない。本研究はこの二つの量を全ネットワーク深度および生成時点にわたって測定することで、その時空間的解離を可視化した。

### 7.2 Predictive Alignment Is Not Functional Equivalence

Cross-model mappingでも同様である。高い$R^2$、CKA、pair retrievalは、

> *one representation predicts another*

ことを示す。しかし、

> *one representation can replace the other inside the target computation*

ことは示さない。

さらに本研究の$\alpha$ sweepは、高いpredictive scoreを維持しながらactivation varianceがtarget distributionの中心方向へ著しく縮小（center collapse）し得ることを示した。したがって、cross-model causal interventionでは少なくとも、
1. predictive fidelity
2. distributional typicality
3. interventional effect

を別々に測定・報告する必要がある。

### 7.3 Scope of the Causal Claim

本研究から導ける結論は、

> **Probe-accessible locations need not coincide with local causal bottlenecks under matched activation substitution.**

である。一方、

> **Affect information has no causal role in report generation.**

とは結論しない。後者を検証するには、系列全体にわたる動的計算や注意機構を介した広域回路の追跡が必要である。したがって、本研究における因果的影響力の評価は、検証した特定の局所スライス介入族に対する操作的知見として解釈されるべきである。

---

## 8. Limitations and High-Value Future Directions

本研究の境界づけと限界は以下の通りであり、これらは健全な今後の拡張課題を構成する：

1. **非局所的・複数層パスパッチングの未網羅**: 単一層全層スイープおよび局所ブロック置換を実施したが、非連続な疎結合回路（sparse circuit）や特定Attention Head間の相互作用パスを網羅するPath Patchingまでは実施していない。
2. **非線形アライメントの未検証**: Base/Instruct間の表現対応づけにはRidge回帰を用いており、非線形多様体アライメントにおける幾何学的歪みの完全な解消には至っていない。
3. **モデル規模**: 検証は1B〜1.5B規模のオープンウェイトモデル（Qwen2.5-1.5B, Llama-3.2-1B）に集中しており、7B以上の大規模モデルにおけるスケール効果の確認は将来の課題である。
4. **因果スイープにおけるサンプル規模の制約**: 全層×3コンポーネントに及ぶ全層網羅因果スクリーニング（Joint OT、生成時パッチング、プローブ方向射影消去）は、計算コスト制約から事前に固定したテストペアのサブセット（15 pairs）を用いて評価された（*The full-layer causal sweep used a computationally constrained subset of 15 held-out matched pairs; therefore, recovery estimates should be interpreted as localization evidence rather than precise population-level effect estimates.*）。なお、代表層（L10, 14, 15, 18, 20, 24）についてはHeld-out完全一致ペア（全39組評価; Prompt有効32ペア, Generation有効39ペア）による全数コホート評価を実施し、中盤層プローブピークにおけるPrompt-time局所因果回復の低さ（~0%）および生成時における後段Residual Streamへの因果レバレッジの動的シフト（最大53.24%）を確認している（代表層には探索スクリーニング優先層が含まれるため、本全数評価は探索効果の安定化と効果量推定を目的とする）。

---

## 9. Conclusion

本研究は、大規模言語モデルにおける情動関連表現をケーススタディとして、内部表現の線形解読能（decodability）と、検証した局所介入における因果的レバレッジ（causal leverage）の所在との乖離を示した。

Qwen2.5-1.5Bを用いた体系的実験と、Llama-3.2-1B-Instructを用いたgeneration-time部分追試により、以下の包括的結論が得られた：

1. 中間層で高いdecodabilityを示し、affective peak-versus-neutral condition の線形デコード能はMLPではLayer 15（$R^2 = 0.5610$）、AttentionではLayer 18（$R^2 = 0.5495$）、Residual streamではLayer 14（$R^2 = 0.5016$）において最大となった。
2. しかし、同一部位の局所活性化を置換しても、自己報告分布の回復率はプロンプト時でわずか 0.51%（Layer 15 MLP）、生成時でも -0.06% に留まり、線形解読能がピークとなる部位に対する matched activation substitution は自己報告分布を実質的に回復しなかった（低局所回復）。
3. 学習済みプローブ方向を除去しても系統的な中和傾向は認められず、その出力変位は直交ランダム方向の除去と統計的に区別されなかった（N=100高解像度追試で全15条件 empirical $p \ge 0.297$, BH-FDR $q = 1.000$）。
4. 一方で、自己報告生成直前のコンテキストにおいては、後段Residual Stream（Layer 18〜24）に強い因果レバレッジが出現し、最大 53.24%（中央値 61.57%）の因果回復を達成した。

総じて、本研究の最終知見は次のように集約される：
> **Affective condition was most linearly decodable at intermediate layers, yet those same sites showed little causal recovery under matched activation substitution. Strong causal leverage instead emerged later, during report generation, particularly in the late residual stream. Probe-aligned direction removal showed no specific local necessity. Together, these results show that linear decodability identifies where information is accessible, but does not by itself identify where, when, or along which direction that information becomes causally effective.**

すなわち、本研究が実証した最終の中心命題は次のように定式化される：

$$\boxed{\text{Linear Decodability} \not\Rightarrow \text{Causal Localization}}$$

$$\boxed{\text{where information is decodable} \neq \text{where/when it becomes causally effective}}$$

より一般的な解釈性研究の文脈において、本知見は次のように総括される：
> **“Linear decodability identifies where information is accessible, but does not by itself identify where, when, or along which direction that information becomes causally effective.”**

したがって、LLM内部表現をmechanisticに解釈する際には、probe accuracyやrepresentation similarityだけではなく、distributional diagnosticsと、prompt時および生成時双方を網羅したproperly validated causal interventionsを独立に組み合わせる必要がある。

---

## 10. Reproducibility & Artifact Mapping

主要結果と実装スクリプト・生成成果物の対応：
- **主図 Figure 1 生成モジュール**: `v3/scripts/plot_main_figure1.py` $\rightarrow$ `v3/results/figure1_four_panel_dissociation.png` / `.pdf`
- **代表6層コホート因果スイープ**: `v3/scripts/run_focused_39pairs_sweep.py` $\rightarrow$ `v3/results/focused_causal_sweep_39pairs.csv`
- **高解像度必要性検定 (N=100)**: `v3/scripts/run_focused_necessity_n100.py` $\rightarrow$ `v3/results/focused_necessity_sweep_n100.csv`
- **生成時多層Residualパッチング**: `v3/scripts/run_generation_multilayer_residual.py` $\rightarrow$ `v3/results/generation_multilayer_residual_results.csv`
- **ブートストラップ95%信頼区間計算**: `v3/scripts/compute_bootstrap_ci.py` ($N_{\mathrm{boot}}=2000$, seed=42; 入力: `focused_causal_sweep_39pairs_pair_level.csv`, `generation_multilayer_residual_results.csv`)
- **全層探索因果局所化スイープ**: `v3/scripts/run_causal_localization_sweep.py` $\rightarrow$ `v3/results/causal_localization_sweep_joint_ot.csv`
- **生成時因果スイープ (Qwen / Llama)**: `v3/scripts/run_generation_time_causal_sweep.py` $\rightarrow$ `v3/results/generation_time_causal_sweep.csv`, `..._llama.csv`

