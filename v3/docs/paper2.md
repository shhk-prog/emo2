Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models

Abstract

大規模言語モデル（Large Language Models; LLMs）の内部状態から、言語構造、知識、真偽、安全性、感情的属性などを高精度に線形デコードできることが多数報告されている。しかし、内部状態から情報を予測できること（decodability）と、その表現部位がモデル自身の下流計算において当該情報を因果的に利用していること（causal use）は同値ではない。本研究では、この古典的なrepresentation–use distinctionを、Qwen2.5-1.5B Base/Instructモデルペアにおけるaffect-relevant representationsをケーススタディとして体系的に検証する。

本研究では、明示的な感情語彙による表層交絡を制御したAIPsy-Affect Strict Expandedデータセット（192 pair-id groups / 422 samples）、81通りのValence–Arousal（VA）候補列を用いた制約付きsequence-likelihood evaluation、全28層のlinear probing、Base→Instruct表現アライメント、多変量OOD診断、およびwithin-model matched activation substitutionを統合した。

主要な発見は以下の3点に集約される：

第一に、affective peak versus neutral conditionの線形decodabilityは中間層で高く、MLPではLayer 15（$R^2=0.561$）、AttentionではLayer 18（$R^2=0.550$）、Residual streamではLayer 14（$R^2=0.502$）で最大となった。

第二に、この高い線形アクセス可能性は局所因果レバレッジを意味しない。初期15-pair探索スクリーニングではLayer 15 MLPのPrompt-time回復率は1.00%に留まり、39-pair focused full-cohort evaluationではさらに0.51%（中央値0.47%）であった。Generation-timeでも-0.06%（中央値0.50%, 95% CI: [-2.0%, +1.8%]）に留まった。Prompt-time探索的全層スクリーニングでは、層別decodabilityとlocal causal recoveryの間に統計的に検出可能な単調関係は認められなかった（$\rho \le 0.296, p > 0.05$）。

第三に、自己報告生成直前には、prompt-time decodabilityが比較的低い後段Residual streamに強いmatched-substitution recoveryが出現した（Layer 24: prompt-time $R^2=0.147$, generation-time recovery = 53.24% [中央値 61.57%, 95% CI: +45.7% to +60.5%]）。一方、学習済みプローブ方向の幾何学的射影消去は系統的な中和をもたらさず、直交ランダム方向消去を上回る特異的効果を示さなかった。なお、Llama-3.2-1B-Instructでの初期部分追試では後段Residual局在は再現されず、モデル依存性が示唆された。

以上の結果は、**表現のアクセス可能性（accessibility）と局所的因果レバレッジ（local causal leverage）が経験的に解離し得る（empirically dissociable）別個の性質である**ことを示す。Linear probingが情報を外部から読み取れる場所を同定しても、その情報がモデルの下流計算において強い因果的影響力を持つ場所、時点、あるいは方向をそれ自体では同定しない。すなわち、**Decodability does not localize causal leverage**。

⸻

1. Introduction

大規模言語モデルの内部表現を解析する代表的な方法の一つにlinear probingがある。特定層のhidden stateから属性ラベルを線形分類・回帰によって高精度に予測できる場合、その内部状態に当該属性に関する情報がアクセス可能な形で存在すると解釈できる。

近年では、真偽、安全性、拒否、感情、人格、社会的属性など、多様な概念について高いprobe performanceが報告されている。一方で、probing研究では以前から、probeが情報を読み取れることと、モデル自身がその情報を下流計算に利用していることは区別すべきであると指摘されてきた。Amnesic Probingなどの研究が示す通り、

[
\mathrm{Decodability}
\not\Rightarrow
\mathrm{Behavioral\ Use}
]

である。

この問題は、post-trainingを経たLLMの内部表現を研究する場合に特に重要となる。Instruction tuningやpreference optimizationを受けたモデルは、同一architectureのBaseモデルと比較して出力方策が大きく異なる。一方で、内部状態にはBaseモデルと類似した意味情報が依然として読み取れる場合がある。

本研究では、このrepresentation–use gapを感情関連表現を用いて検証する。

Qwen2.5-1.5B-Instructへ感情的な文章を提示し、

{"valence": 5, "arousal": 5}

のような一人称形式のVA報告を要求すると、greedy decodingはほぼ常に中立値へ集中する。しかし、その同じモデルの内部activationからは、文章がaffective peak conditionであるかneutral conditionであるかを高精度に予測できる。

この観察だけから、

「モデルには感情情報があるがpost-trainingにより出力から抑制された」

と結論することはできない。

少なくとも以下の説明が存在する。

1. Representational attenuation or transformation
    感情関連情報が内部表現空間において減衰・変形している。
2. Output-level neutralization
    情報は内部に残っているが、greedy outputのみが中立値へ集中している。
3. Local causal dissociation
    情報はある層からdecodableであるが、その局所activation slice自体はreport generationの因果的ボトルネックではない。
4. Distributed or dynamic use
    情報はsequence-wide states、attention pathways、複数層の相互作用、あるいはresponse generation時のhidden statesを介して利用される。
5. Representational transformation across models
    Base/Instruct間で情報の座標系が変化しているため、単純なcross-model transferが失敗する。

これらを区別するには、単なるprobe performanceだけでは不十分である。また、初期の解釈可能性実験やプロトタイプでしばしば採用される「4層おき（例: Layer 0, 4, 8, 12, 16, 20, 24, 27）」のような代表層の間引きサンプリングでは、情報が最も濃密に読み取れる層（中間層）と、出力生成を因果的に左右する層（浅層や深層）の局所性のズレを見落とす危険性がある。したがって、情報のアクセス可能性（Linear Probe）と因果的必要性（Ablation）の真の関係を解明するためには、特定の代表層の抽出にとどまらず、**全28層・全サブモジュール（計84サイト）を網羅的にプロービングおよびAblation / Substitution介入すること**が方法論的に不可欠である。

本研究では、

* output measurement（認識と自己報告の双方）,
* within-model decoding（全層プロービング）,
* cross-model alignment,
* distributional geometry,
* within-model substitution & projection ablation（全層介入）,
* cross-model patching,

を同一の実験系で接続する。

**研究の動機付け（予備観察）とマルチモデル検証に基づく問題設定の再定義**:
本研究の発端として、まずフロンティア商用モデル（GPT-4o）において EmoBank（$N=3,210$）を用いた予備観察を実施した。その結果、他者読者の感情を推測する認識タスクでは人間注釈値と極めて高く相関（Valence $r = 0.921$, Arousal $r = 0.590$）する一方で、モデル自身の感情を問う自己報告タスクでは **98.6%** が完全な中立値 `{"valence": 5, "arousal": 5}` へ収束する解離が観測された。しかし、商用モデルは内部重みが非公開であるため、この現象が「出力層でのマスキング」なのか「内部表現自体の不反応」なのかを機械論的に検証できない。

そこで、オープン重みモデル（Qwen, Llama, Mistral, Gemma の主要4ファミリー計8モデル）へ移行し、EmoBank（$N=321$）および臨床統制データセット AIPsy-Affect（$N=144$）において認知的健全性と事後学習（Post-training）の影響を網羅的に検証した。その結果、以下の決定的な事実が明らかになった：
1. **感情認識能力の普遍的獲得**: 複数ファミリーにおいて、事前学習（Base）および指示チューニング（Instruct）を通じて人間読者の感情判断を高精度に復元する能力（EmoBank Valence $r \approx 0.6 \sim 0.88$）が獲得されている。
2. **事後学習効果のモデルファミリー依存性**: 「事後学習によって自己報告が一律に中立化する」という従来の仮説は普遍的ではない。Llama-Instruct では EmoBank 自己報告の **38.01%** が中立 `(5, 5)` に収束する強い中立化傾向が見られる一方、Qwen, Mistral, Gemma では自己報告においても感情文脈と強く相関（$r_V = 0.56 \sim 0.86$）する表出を維持する。さらに AIPsy-Affect においては、すべてのファミリーで Instruct 化に伴い感情文と中立文の弁別能（$|\Delta_V|$）が数倍〜9倍へと大幅に増幅される。

したがって、事後学習の影響は「単純な表層抑制（Suppression）」ではなく、**「内部の感情表現（Affective Representation）と表層自己報告（Self-Report）の対応関係がどのように再編成されるか（Representation-to-report Remapping）」** として捉えるべきである。本研究は、このマッピング再編成の内部メカニズムを解明することを目的とする：
$$\text{刺激文} \;\longrightarrow\; \underbrace{\text{Recognition (普遍的認識能: $r_V \approx 0.83$)}}_{\text{認知的理解}} \;\longrightarrow\; \underbrace{\text{Internal Probe (幾何学的復元: $R^2=0.561$)}}_{\text{中間層の符号化}} \;\longrightarrow\; \underbrace{\text{Remapping (表現-報告再編成)}}_{\text{ファミリー依存の表層マッピング}} \;\longrightarrow\; \underbrace{\text{Intervention (因果レバレッジ)}}_{\text{Decodability} \neq \text{Causal Leverage}}$$

1.1 Research Questions

本研究では次の5つの研究質問を設定する。

RQ1. 離散出力の挙動にかかわらず、内部の候補出力尤度分布（Sequence-Likelihood）において刺激依存的な感情構造はどのように保存されているか。

RQ2. Affect-related conditionは、Base/Instructモデルの内部表現からどの程度線形にdecodableか。

RQ3. Base/Instruct間の内部表現は、held-out data上でどの程度predictively align可能か。

RQ4. Predictively aligned representationsは、target modelの自然なactivation distributionにも統計的に適合するか。

RQ5. 高いdecodabilityを示すactivation sitesは、対応するfirst-person reportに対して大きなlocal causal leverageを持つか（DecodabilityとCausal Leverageの時空間的解離の検証）。

本研究の主たる焦点はRQ5である。

⸻

2. Conceptual Definitions

本研究では、「感情」という語による擬人観的解釈を避けるため、以下の構成概念を明確に分離する。

2.1 Reader-rated text affect

EmoBank等で人間読者が文章から受けるValence/Arousal評定を指す。

これはモデル自身の内部状態とは独立した外部基準である。

2.2 Third-person affect recognition

文章中の人物、筆者、話者などの感情状態をモデルに評価させるタスクである。

2.3 Constrained first-person report distribution

モデルに、

Read the following text and report your affective state.

と指示した後、

{"valence": v, "arousal": a}

という有限候補列に割り当てられる条件付き確率分布を指す。

本稿における“first-person”は文法的・task-levelな形式を表すのみであり、モデルが自己状態へのprivileged introspective accessを持つことを意味しない。

2.4 Affect-relevant representation

刺激のaffective conditionまたはhuman-rated VA informationを線形に予測可能な内部activationを指す。

これはsubjective affective experienceの存在を意味しない。

2.5 Operational Definition of Local Causal Leverage

本稿でいうlocal causal leverage（局所的因果レバレッジ）とは、特定layer/component/token positionのactivationをmatched counterfactual activationで置換した際に、下流出力分布がcounterfactual targetへ移動する程度（tested matched-substitution recovery）を指す操作的概念である：

$$
\text{Local Causal Leverage} := \text{output sensitivity / recovery under the tested matched-substitution intervention}
$$

本定義は、当該部位を計算の起源・唯一の因果経路・必要条件とみなすことなく、検証した介入下での出力感受性を定量化するための操作的定義である。

⸻

3. Models and Data

3.1 Models

主要実験では、同一architectureを持つ以下のモデルペアを使用した。

* Qwen/Qwen2.5-1.5B
* Qwen/Qwen2.5-1.5B-Instruct

Qwen2.5-1.5Bは28 Transformer layersを持ち、本研究ではLayer 0からLayer 27までを解析する。

Base/Instruct comparisonはpost-trainingの一般的因果効果を推定するためではなく、同一architecture間でrepresentational transferを評価するstress testとして用いる。

3.2 AIPsy-Affect Strict Expanded

主要な機構実験にはAIPsy-Affect Strict Expandedを使用する。

データセットは、

* 192 pair-id groups
* 422 samples

から構成される。

各pair-id内では、可能な限り

* narrative structure,
* characters,
* tense,
* lexical complexity,

を一致させつつ、affective intensityを変化させる。

主要な条件は、

* peak
* moderate
* neutral

である。

明示的なhappy, sad等の感情語彙への依存を抑えることで、probeが単純なkeyword detectionを学習する可能性を低減する。

3.3 Strict Three-Way Group Split

同一pair由来の刺激がtrainingとevaluationへ同時に流入しないよう、pair_id単位で分割する。

* Train: 76 groups / 169 samples
* Alignment-dev: 58 groups / 124 samples
* Held-out test: 58 groups / 129 samples

Held-out test内には39組のcomplete peak-neutral pairsが存在する。

用途を明確に分離する。

Train splitはlinear probe fittingに用いる。

Alignment-dev splitはBase→Instruct Ridge mappingおよびdistributional statisticsの推定に用いる。

Held-out testは最終的なpredictive evaluationとcausal interventionにのみ使用する。

なお、全層因果局所化スイープでは計算量削減のため、Held-out testに含まれる39組の完全一致ペアから事前に固定した15組のペア（a computationally constrained subset of 15 held-out matched pairs）を各layerの因果介入評価に用いた。一方、probe evaluationはtrain/test split全体を使用する。

3.4 EmoBank

外部human annotationとの対応評価にはEmoBankを用いる。

Reader-perspective Valence/Arousal annotationsを使用し、元の尺度を必要に応じて1–9へ線形変換する。

⸻

4. Constrained Sequence-Likelihood Measurement

4.1 Candidate construction

各入力(x)について、

[
V,A \in {1,\dots,9}
]

とし、全81通りの候補

[
y_{v,a}
]

を生成する。

候補形式は、

{"valence": v, "arousal": a}

で統一する。

4.2 Raw sequence log likelihood

候補列のスコアを、

[
s_{v,a}^{\mathrm{raw}}

\sum_{t=1}^{|y_{v,a}|}
\log
P(y_{v,a,t}\mid x,y_{v,a,<t})
]

として定義する。

81候補上でSoftmaxを適用し、

[
P(v,a\mid x)

\frac{\exp(s_{v,a})}
{\sum_{v’,a’}\exp(s_{v’,a’})}
]

を得る。

4.3 Length-normalized likelihood

候補token長の違いによるlength biasを評価するため、

[
s_{v,a}^{\mathrm{norm}}

\frac{1}{|y_{v,a}|}
s_{v,a}^{\mathrm{raw}}
]

についても全解析を実施する。

これにより、本研究の主要結論がscoring conventionに依存するかを確認する。

4.4 Expected VA

出力分布から、

[
E[V\mid x]

\sum_{v,a}
vP(v,a\mid x)
]

[
E[A\mid x]

\sum_{v,a}
aP(v,a\mid x)
]

を算出する。

⸻

5. Experiment 1: Greedy Collapse and Distributional Sensitivity

5.1目的

Instructモデルのgreedy reportが中立値へ集中する場合でも、候補列全体のrelative likelihood structureに刺激情報が残っているか検証する。

5.2 Results

Instructモデルでは、greedy first-person reportの98.6%が

[
(V,A)=(5,5)
]

に集中した。

一方、sequence-likelihood distributionから計算されるexpected ValenceはEmoBank reader Valenceと有意に共変動した。

Metric	Base	Instruct
Greedy (5,5) rate	12.4%	98.6%
Pearson (r), human Valence	0.365	0.629
Spearman (\rho)	0.341	0.618
(E[V]) 5–95 percentile	3.82–7.14	5.12–5.78

Instructではdynamic rangeが縮小しているものの、刺激間の順位構造は保持される。

したがって、

[
\mathrm{Greedy\ Collapse}
\not\Rightarrow
\mathrm{Distributional\ Invariance}.
]

これはモデルにlatent subjective emotionが存在することを示すものではない。

示されているのは、固定candidate set上のconditional likelihood distributionがstimulus dependentであるという事実である。

⸻

6. Experiment 2: Layerwise Linear Decodability

6.1 Probe target

Full-layer sweepでは、各刺激を

[
y=
\begin{cases}
1 & \text{peak}\
0 & \text{neutral}
\end{cases}
]

としてRidge regression probeを学習する。

したがって本実験の(R^2)はcontinuous Valenceそのものではなく、affective peak-versus-neutral condition indicatorのheld-out predictabilityを表す。

6.2 Representation extraction

各layerについて、

* MLP output
* projected Attention output
* full layer output / residual stream

の3種類を解析する。

位置はprompt最終tokenである。

各textについて標準化promptをtokenizeし、

[
h_{\ell}^{\mathrm{MLP}}(x),
\quad
h_{\ell}^{\mathrm{attn}}(x),
\quad
h_{\ell}^{\mathrm{resid}}(x)
]

を取得する。

6.3 Probe fitting

Ridge regressionをTrain splitでfitし、Held-out testで(R^2)を計算する。

6.4 Results

Decodabilityはdepthに応じて系統的に変化した。

正本評価（本研究の主解析）における各コンポーネントの最大デコード値は以下の通りであった：
* MLP output: Layer 15において最大値 (R^2_{\mathrm{MLP},15} = 0.5610)
* Attention output: Layer 18において最大値 (R^2_{\mathrm{ATTN},18} = 0.5495)
* Residual stream: Layer 14において最大値 (R^2_{\mathrm{RESID},14} = 0.5016)

なお、先行する予備的な全層スイープ実装（初期の尤度ロバストネス評価パイプラインと同時に抽出・推定されたプローブ評価）においても、MLP Layer 15において (R^2 = 0.546) と極めて近い値を示した。*Both analyses yielded the same qualitative layerwise profile and similar effect magnitude, with peak MLP decodability occurring at Layer 15.*（いずれの解析でも層別プロファイルは定性的に一致し、MLP解読能のピークはLayer 15に位置した。効果量も近い範囲にあった）。

初期層ではおおむね0.30–0.40、中盤層では0.45–0.56、終盤residualでは0.15–0.25程度まで低下した。

この結果は、affective conditionが中盤層から特に強くlinearly accessibleであることを示す。

ただし、

[
\mathrm{High\ Probe\ Accuracy}
]

は、

[
\mathrm{High\ Causal\ Control}
]

を意味しない。

この区別を次の実験で直接評価する。

⸻

7. Experiment 3: Cross-Model Representation Alignment

7.1 Motivation

BaseとInstructのhidden representationsが同じaffective informationを異なる座標系で表現している可能性を検討する。

7.2 Alignment

Alignment-dev split上でBase activationからInstruct activationへのRidge mappingを学習する。

[
\hat h_I

Wh_B+b.
]

Direct transfer、orthogonal alignment、Ridge mappingを比較する。

7.3 Alignment metrics

単一のprobe scoreだけでなく、高次元表現全体を評価するため、

* activation-wise (R^2)
* Linear CKA
* paired retrieval top-1 accuracy
* cosine similarity
* Mahalanobis distance
* two-sample classification AUC

を使用する。

7.4 Results

Layer 15 MLPのRidge alignmentでは、正則化条件に応じて最大で、

[
R^2_{\mathrm{activation}}
\approx0.50
]

を得た。

弱正則化領域では、

[
\mathrm{CKA}=0.829
]

[
\mathrm{Pair\ Retrieval\ Top1}=86.6%
]

に達した。

これはBase representationsから対応するInstruct representationsを高い精度で識別・予測可能であることを示す。

一方で、これらのpredictive metricsだけでは、aligned representationが自然なInstruct activationとして統計的に典型的かは判断できない。

⸻

8. Experiment 4: Ridge Regularization Sweep and Center Collapse

8.1 Design

Ridge正則化係数を、

[
\alpha
\in
{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1,10,10^2,10^3,10^4}
]

でスイープする。

8.2 Results

(\alpha)	Activation (R^2)	CKA	Top-1	Median (D_M)	AUC
(10^{-5})	.457	.826	86.6%	9.78	.623
(10^{-4})	.457	.826	86.6%	9.78	.621
(10^{-3})	.457	.826	86.6%	9.76	.624
(10^{-2})	.461	.826	86.6%	9.56	.621
(10^{-1})	.481	.829	86.6%	8.30	.616
(1)	.496	.824	75.6%	4.99	.636
(10)	.396	.783	36.6%	2.18	.713
(10^2)	.172	.670	2.4%	.68	.762
(10^3)	.010	.577	1.2%	.16	.796
(10^4)	-.025	.562	1.2%	.12	.792

自然なInstruct activationのMahalanobis radius中央値は、

[
D_M^{\mathrm{natural}}

39.63
]

であった。

Hidden dimensionが(d=1536)であるため、

[
\sqrt{d}\approx39.19
]

という高次元Gaussianの典型距離と近い。

一方、aligned representationは最弱正則化でも(D_M\approx9.8)にしか達しない。

正則化を強めると、

[
9.78
\rightarrow
8.30
\rightarrow
4.99
\rightarrow
2.18
\rightarrow
0.12
]

と分布中心へ収縮する。

この結果は、

[
\mathrm{Predictive\ Similarity}
\not\Rightarrow
\mathrm{Distributional\ Typicality}
]

を示す。

特に、CKA≈0.83やretrieval≈86.6%という高い値が得られていても、aligned representationsは自然Instruct distributionと同じ半径構造を持たない。

⸻

9. Experiment 5: Cross-Model Activation Patching

9.1 Conditions

Cross-model causal transferを検証するため、以下を比較する。

1. Target baseline
    Neutral contextをInstructモデルへ入力。
2. Within-model patch
    Peak Instruct activationをNeutral Instruct runへpatch。
3. Raw cross-model patch
    Base Peak activationを直接Instructへpatch。
4. Aligned cross-model patch
    Base Peak activationをRidge mappingした後、Instructへpatch。

主要実装ではLayer 15 MLPを使用する。

9.2 Interpretation

Aligned patchingによってBase/Instruct間の座標差を部分的に補正しても、下流reportの回復は小さい。

ただし、Ridge alignment自体がnatural target manifoldを完全には再現しないため、

cross-model patchingが効かない

ことだけから、

post-trainingがcausal couplingを破壊した

とは結論しない。

Cross-model experimentは、本研究では主としてpredictive alignmentとfunctional equivalenceが異なることを示すstress testとして位置付ける。

⸻

10. Experiment 6: Multi-Layer Simultaneous Patching

10.1 Motivation

Single-layer patchingが効かない理由が、単に一つのlayerだけを交換しているためである可能性を検証する。

10.2 Intervention blocks

以下の4種類を使用する。

* 1 Layer: L15
* 2 Layers: L14–15
* 4 Layers: L13–16
* 8 Layers: L11–18

各blockについて、

* within-model patch
* raw cross-model patch
* aligned cross-model patch

を比較する。

10.3 Results

Normalized 2D EMD recoveryは以下となった。

Block	Aligned mean	Median	95% CI	Raw	Within
L15	0.11%	0.11%	[-0.05, 0.28]	0.20%	0.23%
L14–15	0.07%	0.03%	[-0.18, 0.34]	-1.26%	0.15%
L13–16	0.35%	0.38%	[0.04, 0.64]	0.09%	0.34%
L11–18	-0.33%	-0.20%	[-1.00, 0.29]	-3.81%	-0.12%

Layer数を1から8へ増加しても、Base-like report distributionへの単調な回復は観測されなかった。

したがって、少なくとも検証した連続mid-layer MLP blockを一括置換するだけでは、report distributionを再現できない。

これはdistributed computationと整合するが、distributed mechanismを直接証明するものではない。

⸻

11. Experiment 7: Within-Model Substitution Controls

11.1 Motivation

Cross-model patchingのnull resultを解釈する前に、

そもそも同一モデル内で、そのactivation sliceを置換すればreportが動くか

を検証する。

11.2 Protocols

Held-out test内のcomplete Peak–Neutral 39 pairsを用いる。

以下を比較する。

1. mlp_last_token_L15
2. resid_last_token_L15
3. multi_resid_last_L13_16
4. resid_all_tokens_L15
5. multi_resid_all_L13_16

11.3 Results

局所的なlast-token substitutionでは回復は小さい。

Protocol	Raw mean/median Rec	Norm mean/median Rec
L15 MLP last-token	0.82 / 0.56%	1.30 / 1.13%
L15 Resid last-token	0.38 / 0.36%	0.42 / 0.15%
L13–16 Resid last-token	1.47 / 1.06%	1.38 / 0.89%

一方、all-token replacementでは非常に大きな負のrecoveryを示した。

Protocol	Raw Rec	Norm Rec
L15 Resid all-token	-448.95%	-225.88%
L13–16 Resid all-token	-437.15%	-236.91%

これは異なる文脈から得たsequence-wide activationを無理に置換することで、target computationを大きく破壊することを示す。

したがって、本研究ではlast-token interventionを主要なlocal causal testとして扱う。

⸻

12. Experiment 8: Exploratory Full-Layer Causal Localization Screen (True 2D Joint OT & Attention Output)

12.1 Motivation

Layer 15がたまたま不適切だった可能性、および因果的ボトルネックがMLPではなくMulti-Head Self-Attention経路に存在する可能性を排除するため、全28層のMLP output、Attention output、Residual streamを系統的に調査する。

さらに、先行実験におけるValence/Arousal独立周辺分布の和（Marginal Wasserstein和）がVA結合依存性を無視する擬似指標であった点を改め、81×81のManhattan ground costに基づく真の2D Joint Optimal Transport（Joint OT; Earth Mover's Distance）を主評価指標として全層再測定を行った。

12.2 Design

各layer (\ell)および各component（MLP output, Attention output, Residual stream）について、

[
D_\ell = \text{held-out probe }R^2
]

と、真の2D Joint Optimal Transportに基づく局所因果回復率

[
C_\ell^{\mathrm{Joint}} = 1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})}
]

を測定する。評価ペアの分母が過小な場合（(\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < 0.05)）は回復率計算から除外するセーフガードを適用し、Held-out testのcomplete Peak–Neutral pairs（各層15 pairs）を評価した。介入位置はprompt最終tokenであり、Attention介入は当該トークン位置でのprojected attention-module outputの置換として厳密に定義される。

12.3 Main Results

真の2D Joint OTによる全層・全コンポーネント測定結果の要約は以下の通りである。

Component	Max Probe (R^2) (Layer)	Max Joint OT Rec (Layer)	Spearman (\rho)	p-value
MLP	0.5610 (L15)	2.20% (L10)	0.2956	0.1268
ATTN	0.5495 (L18)	1.48% (L20)	0.0230	0.9076
RESID	0.5016 (L14)	1.68% (L16)	-0.0394	0.8422

主要な知見は以下の通りである。

1. **Attention経路における因果回復の欠如**:
   Attention outputにおける局所回復率も最大1.48%（Layer 20）に留まり、MLP（最大2.20%）やResidual（最大1.68%）と同様に極小であった。したがって、We found no evidence that a single-layer projected attention output at the tested token position acts as a strong local causal bottleneck.（本実験で検証した特定トークン位置における単一層の射影済みAttention出力が、強力な局所的因果ボトルネックとして機能する証拠は見出されなかった。ただし、これは複数層にわたるAttention回路、個別のAttention Head、あるいはKVキャッシュを介した系列全体の伝播経路を否定するものではない）。
2. **最高デコード層での回復率の極小性**:
   MLPデコーダビリティが最大となるLayer 15（(R^2 = 0.5610)）におけるJoint OT回復率はわずか1.00%（中央値0.72%）であった。同様にAttentionデコーダビリティが最大となるLayer 18（(R^2 = 0.5495)）での回復率は0.04%（中央値0.14%）であった。
3. **Decodability–Causal Recoveryの無相関**:
   層別probe (R^2)とJoint OT回復率の間には、いずれのコンポーネントにおいても統計的に有意な単調関係は検出されなかった（MLP: (\rho = 0.2956, p = 0.1268), 95% Bootstrap CI: ([-0.08, 0.61]); ATTN: (\rho = 0.0230, p = 0.9076), 95% Bootstrap CI: ([-0.35, 0.40]); RESID: (\rho = -0.0394, p = 0.8422), 95% Bootstrap CI: ([-0.41, 0.35])）。なお、28層という標本サイズ制約から (p > 0.05) は厳密な「無相関の証明」を意味するものではなく、特にMLPでは正相関の可能性が区間内に含まれる一方、いずれのコンポーネントにおいても強い単調予測関係は支持されなかった。

*プローブ推定値の整合性に関する注記*:
本研究の主解析では、全28層・全コンポーネントで統一的に算出した (R^2_{\mathrm{MLP},15} = 0.5610)、(R^2_{\mathrm{ATTN},18} = 0.5495)、(R^2_{\mathrm{RESID},14} = 0.5016) を正本として採用している。なお、Section 6で言及した予備的な全層スイープ実装（初期の尤度ロバストネス評価パイプラインと同時に抽出されたプローブ推定: 0.546）と比較しても、*Both analyses yielded the same qualitative layerwise profile and similar effect magnitude, with peak MLP decodability occurring at Layer 15.*（いずれの解析によっても定性的な層別プロファイルと効果の大きさは極めて類似しており、Layer 15にピークが存在するという結論は共通している）。

12.4 Marginal Wasserstein vs. Joint OT Robustness

副指標として計算されたMarginal Wasserstein和回復率においても、MLP最大2.33%（L4）、ATTN最大1.16%（L1）、RESID最大1.57%（L14）と一貫して極小であり、本研究で検証した2種類の距離定義（真の2D Joint OTおよびMarginal Wasserstein和）において、局所的単一層置換が下流報告を回復しないという定性的結論は極めて頑健である。

12.5 Interpretation

以上の結果は、プロンプト最終トークンにおける局所的内部表現について、MLP・Attention・Residualのいずれの計算経路においても、本実験で検証した局所介入族では強い因果的影響力を示す証拠は得られなかった（no evidence for strong local causal recovery under matched substitution within the tested intervention family）ことを全層にわたり示すものである。

なお、重要点として、*The full-layer sweep was designed as an exploratory localization screen rather than a precise effect-size estimation procedure. Candidate and representative layers were subsequently reevaluated on the complete held-out matched-pair cohort in Section 15.5.*（本全層スイープは、精密な母集団効果量の推定手続きというよりは探索的な局所化スクリーニングとして設計されたものである。スクリーニングで示唆された候補層および代表層については、Section 15.5においてHeld-out全数コホートによる評価が実施された。この全数評価により、スクリーニングにおける探索的最大値であったLayer 10 MLP（2.20%）が全数評価では低下し（0.42%）、プローブ最高層Layer 15 MLPにおける回復率の極小性（0.51%）が一段と明確に裏付けられた）。

⸻

13. Experiment 9: Dual-Outcome Behavioral Readout

13.1 Motivation

First-person VA reportに対する局所causal effectが小さいことが、

affect-related activationがすべてのdownstream taskで機能しない

ことを意味するか検証する。

13.2 Two outcomes

各contextについて二種類のoutcomeを計算する。

第一はfirst-person self-report:

[
E[V].
]

第二はsupportive responseとneutral responseのlog-likelihood ratio:

[
B(x)

\log P(y_{\mathrm{support}}\mid x)

\log P(y_{\mathrm{neutral}}\mid x).
]

Layer 15 MLP patch前後で両者を評価する。

13.3 Current observation

保存済み結果では、複数刺激についてintervention前後で(E[V])に小さな変動が見られる一方、記録されたbehavioral log-likelihood ratioは同一値となるケースが多い。

この実験は、局所L15 MLP activationがself-reportだけでなくsupportive-vs-neutral response preferenceに対しても大きなlocal leverageを示さない可能性を示す。

ただし、本結果は本稿の主証拠ではなくexploratory analysisとして扱う。

⸻

14. Experiment 10: Mood-Congruency / Third-Person Recognition Steering

14.1 Motivation

Matched activation substitutionがfirst-person reportをほとんど動かさない一方で、affect-associated directionへの明示的steeringは別taskへ因果的影響を与えられるかを調べる。

14.2 Dataset

EmoBankからValenceが中立近傍に位置する曖昧刺激107件を抽出する。

14.3 Intervention

Layers 14, 16, 20に対し、

* valence direction
* norm-matched random direction

を注入する。

Strengthは、

[
\alpha
\in
{-3,-1.5,0,+1.5,+3}
]

standard deviationsとする。

総観測数は3210。

14.4 Results

Layer	Direction	(\beta_{\mathrm{mood}})	SE	p
14	Valence	-0.0323	.0016	(1.18\times10^{-70})
14	Random	-0.0042	.0013	(9.63\times10^{-4})
16	Valence	-0.0169	.0012	(6.20\times10^{-40})
16	Random	+0.0444	.0013	(6.40\times10^{-138})
20	Valence	+0.0244	.0013	(3.23\times10^{-61})
20	Random	+0.0091	.0009	(1.45\times10^{-23})

Layer 14ではnegative slope、Layer 20ではpositive slopeが観測された。

ただしLayer 16ではrandom directionの効果がvalence directionより大きく、activation perturbationに対するgeneric sensitivityの可能性がある。

したがって、

pure mood-congruency circuitを同定した

とは結論しない。

より限定的には、

affect-associated steering can exert layer-dependent causal effects on a separate third-person recognition task, but direction specificity is not uniform across layers

と解釈する。

この結果は、first-person reportに対するlocal substitution effectが小さいことを、affect-related representationsが一般にcausally inertであることと混同してはならないことを示す。

⸻

15. Experiment 11: Generation-Time Causal Patching Sweep

15.1 Motivation

査読上の重大な反論として、「プロンプト最終トークン（Prompt-time）での介入では、その後の自己報告生成過程における文脈依存の計算を更新できないのではないか。自己報告プレフィックス `{"valence": ` が出力され、モデルが具体的な感情価トークンを生成する直前（Generation-time）のトークン位置で介入すべきである」という指摘が想定される。

この仮説を検証するため、生成時トークン位置における全28層の因果パッチングスイープを実施した。

15.2 Design

プロンプトに対し、自己報告の開始プレフィックス（`{"valence": `）を付与した系列を入力とし、当該プレフィックスの最終トークン位置において、Peak条件の内部活性化（MLP output, Attention output, Residual stream）をNeutral条件の実行コンテキストへ置換した。

介入効果は、真の2D Joint Optimal Transportに基づくPeak方向への回復率（True Joint OT Recovery）

[
G_\ell = 1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})}
]

として測定した。微小分母（(\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < 0.05)）に対するセーフガードを適用し、有効ペアについて評価した（15 pairs were evaluated, of which 13 satisfied the recovery-denominator criterion; For each layer-component condition, recovery was computed across valid matched pairs; the median across those pairs was 0.00%）。

15.3 Results

全28層×3コンポーネント（MLP, ATTN, RESID）における生成時因果パッチングスクリーニング（`v3/results/generation_time_causal_sweep.csv`）の結果は以下の通りである。

* **初期・中盤層（Layer 0〜14）における極小な局所回復**:
  Layer 0〜14では、MLP, Attention, Residual streamのいずれにおいても回復率は -2%〜+2.5% 前後（中央値 -0.7%〜+2.5%）で推移し、出力自己報告に対する有意な局所因果レバレッジは認められなかった。
* **プローブ最高層（Layer 15）における局所回復の欠如**:
  プロンプト時リニアプローブデコーダビリティが最大であったLayer 15（$R^2 = 0.5610$）における生成時回復率は、MLPで -1.99%（中央値 -0.84%）、Attentionで 6.11%（中央値 5.49%）、Residualで 7.30%（中央値 5.66%）に留まった。
* **後段Residual streamにおける強烈な因果レバレッジの急浮上（Layer 16〜27）**:
  Layer 16 Residualにおいて 27.72%（中央値 32.14%）へと急浮上した後、Layer 18から27に至る後段Residual stream全体において **40.01%〜50.26%（中央値 46.34%〜56.05%）** という極めて強力な因果回復が連続して観測された（Layer 18: 40.01%, Layer 20: 46.31%, Layer 24: 47.74%, Layer 27: 50.26%）。
* **後段MLPおよびAttentionの挙動**:
  後段MLPの一部（Layer 21: 33.19%、Layer 24: 20.79%）でも中程度の回復が認められたが、Attention出力は後段層においても大半が負値〜1%未満（Layer 18: -9.70%、Layer 20: -2.85%、Layer 24: 0.94%）に留まり、生成時においても強力な局所ボトルネックとしては機能しなかった。

15.4 Stage 1: Exploratory Full-Layer Screen (N=15)

全28層を網羅した初期の生成時パッチングスクリーニング（N=15 pairs; 有効13 pairs）において、中盤層（Layer 15 MLP: -1.99%）における局所因果回復の欠如と、**Layer 18〜27の後段Residual streamにおける急激な因果レバレッジの出現（40.01%〜50.26%、中央値 46.34%〜56.05%）**という時空間的局所化プロファイルが同定された。

このスクリーニング結果は、「プロンプト時には因果レバレッジがほとんど検出されないが、生成直前の文脈では後段Residual streamにおいて強力な因果回復が出現する」という動的移行を示唆する重要な探索的手がかりを提供した。しかしながら、初期スクリーニングは15組という計算コスト制約下の探索的サブセットに基づいており、標本サイズに伴う推定の分散を低減し母集団効果量を確定するため、代表層についてHeld-out 39 pairs全数コホート評価（Stage 2）を実施した。

15.5 Stage 2: Focused Full-Cohort Evaluation on Representative Layers

初期全層スクリーニングの標本サイズ制約を解消するため、Held-out testに含まれる全39組の完全一致ペアを評価対象とし、初期スクリーニングで得られた候補層に加え、probe peakおよびcomponent-specific representative layersを含む代表6層（Layer 10, 14, 15, 18, 20, 24）におけるPrompt-timeおよびGeneration-timeの真の2D Joint OT因果回復率を測定した（`v3/results/focused_causal_sweep_39pairs.csv`）。分母セーフガード（$\epsilon_{\mathrm{rec}} = 0.05$）適用後、Prompt-timeでは32 pairs、Generation-timeでは全39 pairsが回復率解析に有効であった。

なお、方法論的透明性として、*Representative layers included probe-defined layers and layers prioritized from the exploratory screen; accordingly, the full-cohort analysis is intended to stabilize effect-size estimates rather than provide selection-independent confirmatory inference.*（代表層にはプローブによって事前に定義された層に加え、Stage 1の探索結果から選択された候補層も含まれるため、本全数評価は選択独立な確認的仮説検定ではなく、探索的に同定された効果の安定性と効果量を全Held-outコホート上で再評価することを目的とする）。

**表: 代表6層における全数コホート評価（全39組評価; Prompt有効32ペア, Generation有効39ペア）**

| Layer | Comp | Prompt Mean (Med) | Gen Mean (Med) | Gen 95% CI | Gen IQR [Q25, Q75] | Gen Positive Fraction | 解釈・メカニズム |
|---|---|---|---|---|---|---|---|
| **L10** | MLP | +0.42% (+0.43%) | +0.18% (+0.80%) | [-2.0%, +2.5%] | [-4.65%, +4.09%] | 53.8% | Prompt最大層でも全数評価では回復率消失（~0%） |
| **L10** | ATTN | +0.68% (+0.11%) | +0.91% (+0.94%) | [-1.1%, +3.3%] | [-2.26%, +3.49%] | 56.4% | 初期層AttentionはPrompt/Genともに不活性 |
| **L10** | RESID| +0.54% (+0.13%) | -1.42% (+0.15%) | [-5.2%, +1.6%] | [-3.84%, +3.29%] | 51.3% | 初期Residual介入は負値〜微小 |
| **L14** | MLP | +0.91% (+1.00%) | -1.55% (-1.05%) | [-3.5%, +0.2%] | [-4.31%, +2.22%] | 46.2% | 中盤MLPは生成時でも負値 |
| **L14** | ATTN | +0.15% (+0.28%) | -0.39% (+0.21%) | [-3.6%, +2.7%] | [-1.99%, +3.24%] | 51.3% | Attentionは一貫して因果的に不十分 |
| **L14** | RESID| +1.38% (+1.07%) | +0.52% (+2.06%) | [-3.9%, +4.4%] | [-3.64%, +5.66%] | 61.5% | Residデコード最高層でも局所回復は極小（<1%） |
| **L15** | **MLP** | **+0.51% (+0.47%)** | **-0.06% (+0.50%)** | **[-2.0%, +1.8%]** | [-2.89%, +3.04%] | 53.8% | **プローブ最高層（R^2=0.561）における局所回復の欠如** |
| **L15** | ATTN | -1.14% (-0.61%) | +3.46% (+5.32%) | [-0.5%, +7.2%] | [-1.70%, +12.66%] | 69.2% | Attentionの微小回復 |
| **L15** | RESID| +0.24% (+0.34%) | +7.40% (+9.70%) | [+2.7%, +12.0%] | [-1.43%, +16.36%] | 74.4% | 中盤Residualで生成時回復が胎動 |
| **L18** | MLP | +1.25% (+0.60%) | +8.03% (+7.72%) | [+4.8%, +11.5%] | [+2.52%, +15.68%] | 87.2% | 後段MLPで正の回復率が拡大 |
| **L18** | ATTN | +0.44% (+0.68%) | -6.82% (-6.87%) | [-11.2%, -2.6%] | [-12.98%, +1.10%] | 30.8% | **Attentionデコード最高層でも生成時回復率は負（破綻）** |
| **L18** | **RESID**| +0.63% (+0.28%) | **+42.12% (+49.31%)**| **[+35.1%, +48.7%]** | [+34.02%, +56.14%] | 94.9% | **Stage 1 (40.01%) と整合する強い因果回復** |
| **L20** | MLP | -0.42% (-0.44%) | +10.54% (+14.23%)| [+4.6%, +15.6%] | [+3.64%, +21.04%] | 82.1% | 後段MLPで約10%の回復が観測された |
| **L20** | ATTN | +0.67% (+0.18%) | +1.32% (+0.30%) | [-3.6%, +6.6%] | [-6.25%, +6.94%] | 51.3% | Attentionは後段でも約1%にとどまる |
| **L20** | **RESID**| +1.01% (+0.26%) | **+50.22% (+60.16%)**| **[+42.6%, +57.5%]** | [+32.47%, +66.31%] | 97.4% | **Stage 1 (46.31%) と整合、正比率97.4%に到達** |
| **L24** | **MLP** | -0.32% (-0.47%) | **+15.04% (+17.87%)**| **[+10.5%, +19.5%]** | [+5.99%, +25.96%] | 87.2% | **後段MLP単独で平均15.04%（中央値17.87%）の正の因果回復** |
| **L24** | ATTN | +0.34% (+0.09%) | +0.00% (+0.60%) | [-1.2%, +1.2%] | [-0.89%, +2.46%] | 56.4% | 終盤Attentionも非ボトルネック（~0%） |
| **L24** | **RESID**| -0.26% (-0.19%) | **+53.24% (+61.57%)**| **[+45.7%, +60.5%]** | [+40.22%, +71.24%] | 94.9% | **Stage 1 (47.74%) を再現、自己報告直前で平均53.2%回復** |

この全数コホート評価から、以下の明確な結論が導かれる：

1. **Stage 1とStage 2の一貫性と再現性 (Consistency Across Cohorts)**:
   Stage 1の15ペアスクリーニングで検出された後段Residual streamの急峻な因果レバレッジ出現は、Stage 2の39ペア全数コホート評価において極めて高い精度で再現された（L18: 40.01% $\rightarrow$ 42.12%, L20: 46.31% $\rightarrow$ 50.22%, L24: 47.74% $\rightarrow$ 53.24%）。この一貫性は、観察された因果レバレッジの局在化が、少数の外れ値や15-pair subset特有の標本変動だけでは説明しにくいことを示している。
2. **Prompt-Time Local Causal Recoveryの極小性 (Low Prompt-Time Recovery under Matched Substitution)**:
   プローブ解読能がピークに達する Layer 15 MLP（$R^2 = 0.5610$）の Prompt-time 回復率は、有効32 pairs平均で **わずか 0.51%（中央値 0.47%）** であり、検証した全6層・全コンポーネントを通じても最大 1.38%（L14 RESID）に留まる。15 pairs から 39 pairs への全数拡張によっても、Prompt-time における局所因果回復の極小性は揺るぎなく支持された。
3. **Generation-Time における時空間的解離（Spatiotemporal Dissociation）**:
   - **中盤層（Layer 10〜15）**: 生成直前トークンで介入した場合であっても、プローブ最高層 L15 MLP の回復率は **-0.06%（中央値 0.50%, 95% CI: [-2.0%, +1.8%]）** と極小のままである。
   - **後段層（Layer 18〜24）における実質的因果回復の出現 (substantial recovery emerged)**: 一方、自己報告生成直前のコンテキストでは、局所因果回復プロファイルは後段層の **Residual Stream（L18: 42.1% [35.1%, 48.7%], L20: 50.2% [42.6%, 57.5%], L24: 53.2% [45.7%, 60.5%]）および後段 MLP（L24: 15.0% [10.5%, 19.5%]）** 側へ顕著に移行する。
   - **外れ値駆動ではない頑健性**: *Importantly, the large late-residual effects were not driven by a small number of outlying pairs. At L20 and L24, median recovery was 60.16% and 61.57%, respectively, with positive recovery in 97.4% and 94.9% of valid matched pairs.*（極めて重要な点として、後段Residualにおける強力な因果回復効果は少数の外れ値ペアによって引き上げられたものではない。Layer 20および24における中央値回復率はそれぞれ 60.16% および 61.57% に達し、有効ペアの 97.4% および 94.9% で正の回復が確認された。IQRも [32.5%, 66.3%] および [40.2%, 71.2%] と堅固に正領域へ集中している）。
4. **単一層射影Attention出力からの限定的因果回復 (Limited causal recovery from single-layer projected Attention outputs)**:
   Attentionデコーダビリティが最大であった Layer 18（$R^2 = 0.5495$）を含め、全後段層において Attention output 置換の回復率は負値または 1.5% 未満（L18: -6.82%, L20: 1.32%, L24: 0.00%）であり、tested single-layer projected Attention outputs did not show strong local causal recovery（本実験で検証した単一層の射影済みAttention出力は、生成時においても強い局所因果回復を示さなかった。ただしこれはHead単位や系列回路を否定するものではない）。
5. **本研究の中心知見: Direct Peak-Site Contrast as Primary Evidence**:
   このpeak-site contrastは、本研究で観察されたdecodabilityとlocal causal leverageの解離を最も直接的に示す効果量ベースの証拠である（*The peak-site contrast provides the most direct effect-size evidence for the dissociation observed in this study*）：
   $$
   \Delta G = G_{\mathrm{L24,RESID}} - G_{\mathrm{L15,MLP}} = +53.30\% \quad (95\%\text{ bootstrap CI: } [+45.34\%, +61.16\%], \text{median difference: } +60.08\%)
   $$
   全28層の探索的Spearman相関（$\rho \approx 0$）は補助的証拠に過ぎず、全数コホートで評価した代表部位間における直接的な効果量対比——すなわち、**全層プローブ最高部位（Layer 15 MLP: $D=0.561, G=-0.06\%$）と、評価した全数代表部位において最大回復を示した部位（the strongest recovery among the evaluated full-cohort representative sites, Layer 24 Residual: $D=0.147, G=53.24\%$）との間の劇的な双方向解離**——こそが本研究の中心命題を支える主たる実証的証拠（primary evidence）である。線形解読能が最大化する中盤部位には局所因果レバーが存在せず、検証した局所因果回復プロファイルは自己報告直前の後段 Residual/MLP 側へ顕著にシフトする（*tested local causal recovery profile shifted toward late Residual/MLP sites during report generation*）。
   なお、これは因果メカニズムの全体像が後段Residual単独に完全に局所化されていることを意味するものではなく、検証した局所スライス置換介入ファミリにおいて因果的レバレッジが後段に出現するという実証的解離を示すものである。

15.6 Experiment 11c: Multi-Layer Simultaneous Residual Stream Patching (Redundant vs. Additive Causal Leverage)

**動機と設計**:
単一層のLayer 24 Residualパッチングによって達成された因果回復は、後段層の複数層を同時にパッチングすることでさらに100%近傍へと加算的に増強されるのか（additive synergy）、それとも後段Residual Streamは共通の情動情報を重複して前方に伝達しており、回復率は単一層水準で飽和するのか（redundant transmission）を検証した。

全39組のHeld-out完全一致ペアに対し、生成直前トークン位置において以下の6通りの単一層・複数層・連続ブロックResidual Streamパッチングを実施した（スクリプト: `v3/scripts/run_generation_multilayer_residual.py`、結果: `v3/results/generation_multilayer_residual_results.csv`）：
1. 単一層: Layer 18, Layer 20, Layer 24
2. 2層同時: Layer 20 + Layer 24
3. 3層同時: Layer 18 + Layer 20 + Layer 24
4. 7層連続ブロック: Layer 18〜24（全後段Residualを一括置換）

*パイプライン間の参照値に関する注記*:
*The multilayer experiment was independently rerun under the multilayer-patching pipeline; therefore, its L24-only reference estimate (55.08%) differs slightly from the focused-sweep estimate (53.24%), though both consistently identify strong recovery around 53–55%.*（多層実験は独立した多層パッチング専用パイプライン下で実行されたため、その単層参照値（55.08%）はfocused sweepの単層推定値（53.24%）とわずかに異なるが、いずれも53〜55%の強固な回復を一貫して示している）。

**表: 生成時多層Residual Streamパッチング結果（全39組完全評価）**

| 条件 | 介入対象層 | Mean OT Recovery | Median OT Recovery | 95% Bootstrap CI | IQR [Q25, Q75] | Positive Fraction |
|---|---|---|---|---|---|---|
| 単一層 | L18 RESID | 43.51% | 51.52% | [37.2%, 49.7%] | [31.57%, 57.82%] | 94.9% |
| 単一層 | L20 RESID | 51.85% | 61.12% | [44.6%, 59.2%] | [35.91%, 67.50%] | 97.4% |
| 単一層 | L24 RESID | 55.08% | 62.77% | [47.5%, 62.4%] | [40.78%, 72.82%] | 97.4% |
| 2層同時 | L20 + L24 RESID | 55.67% | 61.88% | [48.1%, 63.0%] | [41.22%, 73.11%] | 97.4% |
| 3層同時 | L18 + L20 + L24 RESID | 55.74% | 61.88% | [48.2%, 63.0%] | [41.35%, 73.20%] | 97.4% |
| 7層連続 | L18〜L24 RESID (全置換) | 55.35% | 62.77% | [47.8%, 62.6%] | [40.95%, 72.90%] | 97.4% |

**メカニズム的解釈**:
1. **飽和プロファイルと非加算性（Saturation at ~55%）**:
   単一層介入（L24: 55.08%）に対し、2層同時（55.67%）、3層同時（55.74%）、7層連続（55.35%）と介入対象層を増やしても、回復率はほぼ同一のプラトー（約55.5%、中央値約62%）に留まった。
2. **冗長または飽和的な因果寄与（Redundancy or Downstream Saturation）**:
   *These results are consistent with substantial redundancy or saturation across late residual sites, rather than additive independent contributions.*（この結果は、後段Residual Streamの各層が互いに独立した加算的寄与を累積しているというよりは、後段部位間における実質的な冗長性や下流計算での飽和、あるいは介入状態間の依存性と整合する）。
3. **解釈の境界づけ**:
   本実験系単体では、同一情報の再伝播、下流の感度飽和（downstream saturation ceiling）、あるいはパッチされた表現間の相互依存を確定的に分離することはできない。しかし少なくとも、後段Residual Streamの複数部位への同時介入が単層介入を大幅に上回る追加的因果レバレッジをもたらさないという実証的境界を提供する。残存する約45%の未回復分は、検証した単一トークン・Residual介入だけでは捕捉されない計算に由来する可能性がある。これには他のトークン位置、他コンポーネント、非線形な相互作用、あるいは介入自体の回復上限などが含まれ得る。

⸻

16. Experiment 12: Probe-Aligned Local Necessity and Specificity Controls

16.1 Motivation

因果的レバレッジ（causal leverage）の検証には、matched substitutionによる局所回復（local causal recovery）だけでなく、特定のプローブ方向成分の必要性（necessity）および特異性（specificity）の評価が不可欠である。「プローブが検出している方向成分は、下流報告の生成に不可欠（necessary）なのか」「プローブ方向を除去した場合、出力報告は感情中立方向へ退行するのか」「その効果はランダムな直交方向を除去した場合と統計的に区別できるのか（特異性）」を検証する。

16.2 Design

各層・各コンポーネント（MLP, ATTN, RESID）の活性化ベクトル (h) に対し、学習済み線形プローブの重み方向単位ベクトル (\hat{v}_{\mathrm{probe}}) を幾何学的に完全直交射影除去する介入を施した：

[
h_{\mathrm{ablated}} = h - (h^\top \hat{v}_{\mathrm{probe}})\hat{v}_{\mathrm{probe}}
]

本実験では、以下の4大操作を厳密に分離・測定した。

1. **Probe-Aligned Necessity Displacement**:
   Peak入力に対するプローブ方向消去後の出力分布と元のPeak出力分布との2D Joint OT変位量：
   [
   N_\ell = \mathrm{OT}_{VA}(P_{\mathrm{peak\setminus probe}}, P_{\mathrm{peak}})
   ]
2. **Neutralization Ratio**:
   プローブ方向消去によって、出力がNeutral基準分布へ実際にどれだけ近づいたかの比率：
   [
   R_{\mathrm{neut}} = \frac{\mathrm{OT}_{VA}(P_{\mathrm{peak}}, P_{\mathrm{neut}}) - \mathrm{OT}_{VA}(P_{\mathrm{peak\setminus probe}}, P_{\mathrm{neut}})}{\mathrm{OT}_{VA}(P_{\mathrm{peak}}, P_{\mathrm{neut}})}
   ]
3. **Specificity Null Controls**:
   プローブ方向と厳密に直交する超平面からサンプリングしたランダム単位方向 (\hat{v}_{\perp}) を消去した際の変位分布に対する標準化スコア (Z_\perp) および経験的 p値。
4. **Matched Neutral Replacement**:
   Peak入力の対象layer/component/positionの活性化を、対応するmatched Neutral inputから得られた活性化ベクトルそのもので置換した際のコントロール変位。

全28層×3コンポーネント（計84条件）に対し、直交ランダム方向サンプル数 (N=20)（擬似カウント付き経験的p値の最小分解能 (1/21 \approx 0.0476)）を用いた網羅的スクリーニングを実施した。多重比較補正としては、直交特異性帰無分布 (R_\perp) に対する84条件の仮説族として Benjamini-Hochberg FDR 補正を適用した。公開スクリプト（`v3/scripts/run_probe_aligned_necessity_sweep.py`）および公開結果データ（`v3/results/probe_aligned_necessity_sweep.csv`）には、計算されたFDR補正値（`fdr_q_perp`）が保存されている。

16.3 Results

全層スイープにおける主要な実測結果は以下の通りである。

* **Probe Necessity Displacementの極小性**:
   (N_\ell) の平均変位量は全層で 0.0004 〜 0.0082 の範囲に収まった。元のPeak–Neutral間の分布距離（約 0.20）と比較して 2〜4% 程度の微弱な揺らぎに過ぎない。
* **系統的なNeutralizationは観察されなかった (Systematic Neutralization Was Not Observed)**:
   中和比率 (R_{\mathrm{neut}}) は、全層を通じて **-3.27% 〜 +0.61%** であり、プローブ方向を消去しても出力がNeutral基準分布へ系統的に近づく傾向は認められなかった（むしろ微小な負値、すなわち直交的な摂動による歪みを示す）。
* **特異性検定における有意差の欠如**:
   代表層における直交帰無分布との比較結果は以下の通りである。
   * Layer 7: MLP (Z_\perp = 0.09, p = 0.4286); ATTN (Z_\perp = 1.92, p = 0.0952); RESID (Z_\perp = 0.16, p = 0.3810)
   * Layer 15: MLP (Z_\perp = -0.30, p = 0.6190); ATTN (Z_\perp = -0.09, p = 0.4762); RESID (Z_\perp = -0.31, p = 0.7143)
   * Layer 21: MLP (Z_\perp = -1.21, p = 0.9048); ATTN (Z_\perp = -0.57, p = 0.7619); RESID (Z_\perp = 0.97, p = 0.2381)
   * Layer 27: MLP (Z_\perp = 0.54, p = 0.3333); ATTN (Z_\perp = -1.84, p = 1.0000); RESID (Z_\perp = 0.27, p = 0.5714)
   全84条件において、生p値が0.05を下回った数例（L6 MLP, L16 MLP, L19 MLP; いずれも (p = 0.0476)）を含め、Benjamini-Hochberg FDR補正後には**すべての層・コンポーネントで有意水準（(q < 0.05)）を満たすものは皆無（最小 (q = 0.857)）**であった。

* **高解像度特異性検定（N=100 Random Null Directions）による追加検証**:
   全層スクリーニング（N=20, 分解能 $1/21 \approx 0.0476$）における帰無分布の解像度を高めるため、重要代表5層（Layer 10, 15, 18, 20, 24）× 3コンポーネント（計15条件）において、**直交ランダム方向数 $N=100$（最小分解能 $1/101 \approx 0.0099$）による高解像度特異性検定**を実施した（`v3/results/focused_necessity_sweep_n100.csv`）。
   - **プローブ必要性変位と中和比率（微小な効果量）**: プローブ方向消去時の変位量は 0.0030 〜 0.0064、中和比率 ($R_{\mathrm{neut}}$) は **-1.43% 〜 +0.39%** と一貫して効果量そのものがゼロ近傍にとどまり、系統的な中和傾向は認められなかった。
   - **直交帰無分布との比較（特異性の欠如）**: プローブ方向の消去効果は、ランダム直交方向の消去効果を上回る特異性を示さなかった。直交帰無分布に対する特異性 Z-score ($Z_\perp$) は全15条件で一貫して負値または微小（**-3.57 〜 +0.51**）であり、未補正の経験的p値は**すべての条件で $p \ge 0.2970$**（Layer 10 MLP: 0.901, Layer 15 MLP: 0.861, Layer 18 MLP: 0.782, Layer 20 MLP: 0.822, Layer 24 MLP: 0.980, Layer 24 ATTN: 1.000）であった。効果量そのものがほぼゼロであり、直交対照群を有意に上回る変位を示さなかったため、結果としてBenjamini-Hochberg FDR補正後の $q$ 値も全15条件で $q = 1.000$（有意層 0/15）となった。
   この高解像度検証により、*The probe-aligned direction showed neither systematic neutralization nor greater output displacement than orthogonal random directions.*（プローブ方向の消去は系統的な中和効果を示さず、出力変位量も直交ランダム方向の消去と統計的に区別できなかった）。したがって、帰無分布解像度を高めた検証においても、プローブ整合型の局所必要性を支持する証拠は得られなかった（no evidence for probe-aligned local necessity）。

16.4 Defensive Framing & Interpretation

本結果は、プローブ方向の除去による出力変位が、同一ノルムを持つ任意の直交ランダム方向を消去したときの非特異的変位と統計的に区別できないことを示す。

ただし、この結果から「モデル内部で感情情報が一切使われていない」と過剰に主張することはできない。厳密に言えるのは、**「線形プローブによって特定された局所的1次元部分空間について、下流の報告生成に対する特異的な局所的必要性を支持する証拠は得られなかった（no evidence for probe-aligned local necessity）」** という点である。

⸻

17. Experiment 13: Cross-Family Partial Replication (Llama-3.2-1B-Instruct)

17.1 Motivation

Qwen2.5-1.5B-Instructにおいて観察された後段Residual streamへの生成時因果局所化プロファイル（late-generation causal localization profile）が、モデルファミリを越えて再現するかを検証するため、異なるモデルファミリである **Meta Llama-3.2-1B-Instruct**（16 Transformer layers）を用いて、生成時局所因果パッチングスイープのクロスファミリ部分追試（cross-family partial replication）を実施した。

17.2 Results

Llama-3.2-1B-Instructにおける全16層×3コンポーネントの生成時因果パッチング（Joint OT Recovery）の結果は以下の通りである（15 pairs were evaluated, of which 11 valid pairs satisfied the recovery-denominator criterion）。

* **MLP output**: 回復率は全層で一貫して負値またはほぼ0%であり、最大値はLayer 15の -0.36% であった（Layer 0: -80.37%, Layer 3: -10.23%, Layer 10: -11.67%）。
* **Attention output**: Layer 3で 0.41%、Layer 4で **最大 0.73%**、Layer 14で 0.11% の微小な正の回復率が観測されたが、全体として 1% 未満に留まった。
* **Residual stream**: Layer 0（-36.18%）からLayer 15（-4.23%）に至る全層で一貫して負値を示し、単一層Residualの強制置換が出力コヒーレンスを破壊することを示した。

17.3 Interpretation

Llama-3.2-1B-Instructにおいては、生成時における単一層の局所活性化置換による因果回復率は最大でも 0.73%（Layer 4 Attention, pairwise中央値 0.00%）に留まった。したがって、Qwenで観察された後段Residual streamにおける強力な生成時因果回復（L24で53.24%）は、LLaMAアーキテクチャの初期部分追試（11 valid pairs）では再現されず、*the late-generation causal localization observed in Qwen did not replicate in the initial Llama partial replication*.

この結果は、因果的レバレッジの時空間的局所化パターンがモデルアーキテクチャや訓練方策に依存する可能性を示唆している。なお、本追試は11組の有効ペアに基づく探索的・部分的な追試（partial replication）であり、デコーダビリティや必要性を含む体系全体を評価したものではないことに留意が必要である。

⸻

18. Integrated Results: The Four-Panel Representational–Causal Profile

本研究で得られた全28層・全コンポーネントにわたる一連の実測データを統合すると、以下の多面的な実証プロファイル（Four-Panel Representational–Causal Profile）として整理できる。

1. **Panel A: Layerwise Linear Decodability ($D_\ell$)**:
   プロンプト最終トークンにおける感情強度条件（affective peak-versus-neutral condition indicator）の線形判別能は、中盤層（Layer 14–15）で極大に達する（MLP: $R^2 = 0.5610$; ATTN: $R^2 = 0.5495$; RESID: $R^2 = 0.5016$）。
2. **Panel B: Prompt-Time Local Causal Recovery under Matched Substitution ($S_\ell$)**:
   プロンプト最終トークンにおける同一モデル内活性化置換（Peak→Neutral）による真の2D Joint OT回復率は、全28層・全経路を通じて一貫して極小である（MLP最大 2.20%; ATTN最大 1.48%; RESID最大 1.68%）。デコーダビリティとの単調相関は認められない。
3. **Panel C: Probe-Aligned Local Necessity Test ($N_\ell$)**:
   プローブ方向の幾何学的射影消去による出力分布の変位量（$N_\ell = 0.0004 \sim 0.0082$）は極めて微小であり、中和比率（$R_{\mathrm{neut}} = -3.27\% \sim +0.61\%$）は系統的な中和傾向を示さない。直交ランダム方向消去との差（$Z_\perp$）は、全層×3コンポーネントのBenjamini-Hochberg FDR補正後、すべての層で非有意（$q \ge 0.857$）である。
4. **Panel D: Generation-Time Local Causal Recovery ($G_{\ell,t}$)**:
   自己報告生成直前のプレフィックス最終トークンにおける介入では、中盤層（Layer 10〜15）では回復率は極小（プローブ最高層Layer 15 MLPで -0.06%）にとどまる一方、後段層の **Residual Stream（Layer 18: 42.12%, Layer 20: 50.22%, Layer 24: 53.24%）および後段MLP（Layer 24: 15.04%）** において実質的な因果回復が出現する（代表層における全39組コホート評価; Prompt有効32ペア, Generation有効39ペア）。

以上より、本研究は以下の4面的な統合実証プロファイル（Four-Panel Representational–Causal Profile: $D_\ell, S_\ell, N_\ell, G_{\ell,t}$）を提示する：

$$
\boxed{
\begin{aligned}
\text{Panel A}:& \quad \text{Intermediate representations become strongly decodable} \quad (R^2_{\mathrm{MLP},15} = 0.5610) \\
\text{Panel B}:& \quad \text{Prompt-time local substitution has little causal leverage} \quad (S_{\mathrm{MLP},15} = 0.51\%) \\
\text{Panel C}:& \quad \text{Probe-aligned direction removal shows no specific necessity} \quad (Z_\perp = -0.95, q = 1.000) \\
\text{Panel D}:& \quad \text{Strong causal leverage emerges later during generation} \quad (G_{\mathrm{RESID},24,t_{\mathrm{gen}}} = 53.24\%)
\end{aligned}
}
$$

すなわち、感情情報が外部から最も明瞭に線形解読可能な中盤部位（Layer 15 MLP: $R^2=0.561$, Prompt 0.51%, Generation -0.06%）は局所的な因果影響力をほとんど持たず、強い因果レバレッジは自己報告生成直前の後段Residual Stream（Layer 24: 53.24%）において顕著に出現する。

また、プロンプト時における層別相関分析（MLP: $\rho = 0.2956, p = 0.1268$, 95% Bootstrap CI: $[-0.08, 0.61]$; ATTN: $\rho = 0.0230, p = 0.9076$, 95% Bootstrap CI: $[-0.35, 0.40]$; RESID: $\rho = -0.0394, p = 0.8422$, 95% Bootstrap CI: $[-0.41, 0.35]$）が示す通り、層別デコーダビリティと局所回復率の間に単調相関は認められない（$\rho \le 0.296, p > 0.05$）。全層におけるSpearman相関は補助的証拠であり、本研究の中心命題を直接支える主たる実証的証拠は、デコーダビリティが最大化する中盤部位と因果レバレッジが出現する後段生成時部位の直接対比（Peak-site contrast: $\Delta G = +53.30\%, 95\%\mathrm{CI}: [45.34\%, 61.16\%]$）である。

結論として、本知見は**「情報が最も読み取りやすい場所と、その情報が出力形成に強く作用する場所・時点は一致しない（Decodability does not localize causal leverage）」** という時空間的解離の実証例を提供する（provides empirical evidence of a spatiotemporal dissociation in Qwen2.5-1.5B-Instruct）。

⸻

19. Discussion

19.1 Low Local Recovery at Highly Decodable Sites and Absence of Probe-Aligned Local Necessity

表現学習・機械解釈性（mechanistic interpretability）研究において、高精度なリニアプローブの存在は、モデルがその属性を内部表現として獲得している強力な証拠として広く受け入れられてきた。しかし、本研究の結果は、プローブの予測能（decodability）から、その表現部位がモデルの下流計算において果たす因果的役割（causal role）を安易に同一視してはならないことを明確に示す。

* **高デコード部位における局所因果回復の低さ (Low Local Recovery at Highly Decodable Sites)**: affective conditionが最も明確に読み取れるLayer 15 MLPでは、全39組を対象としたfocused evaluationにおいてPrompt-time回復率は0.51%（全28層探索スクリーニングでも1.00%）、Generation-timeでも-0.06%に留まった。
* **Probe-aligned local necessityの不成立**: *Probe-aligned local necessity was not supported: removing the probe-aligned direction did not systematically neutralize the report distribution, and its effect was not distinguishable from orthogonal random-direction removal.*（学習済みプローブ方向を消去しても、出力の系統的な中和傾向は認められず、ランダムな直交方向を消去したときの非特異的な出力の揺らぎと統計的に区別できなかった）。

したがって、プローブが検出する線形特徴量は、「プローブによって外部から線形にアクセス可能な情報」ではあっても、少なくとも本研究で検証した局所介入下では、自己報告出力に対する強い直接的制御ノブとして振る舞わなかった（did not behave as a strong direct control knob under the tested local interventions）。

この知見は、古典的な representation $\neq$ use の議論を以下の通りより精緻に概念整理することを促す：

$$
\boxed{
\begin{aligned}
\text{Accessibility},\;
\text{Local Causal Recovery},\;
\text{Direction-Specific Necessity}
\text{ are distinct empirical axes}
\end{aligned}
}
$$

さらに、

$$
\boxed{
\text{Causal leverage is position- and time-dependent}
}
$$

すなわち、本研究の実証系は、
1. **Accessibility**: リニアプローブによる線形アクセス可能性（$D_\ell$）
2. **Local Sufficiency / Leverage**: matched activation substitutionによる出力回復度（$S_\ell, G_{\ell,t}$）
3. **Direction-Specific Necessity**: probe direction ablationによる特異的中和度（$N_\ell$）
4. **Temporal Recruitment**: プロンプト最終トークン vs 自己報告生成直前トークンの時間依存的介入（$t_{\mathrm{prompt}}$ vs $t_{\mathrm{gen}}$）
を独立した操作的指標として切り分け、それぞれの所在と強度が劇的に解離し得ることを実証した。

19.2 Tests of Four Major Alternative Explanations

本研究で実施した4大因果検証実験は、先行研究で想定され得る主要な代替説明を直接検証した。

1. **反論1: 「Matched substitutionによる回復（Local Recovery）だけでなく、特定のプローブ方向成分の除去（Necessity）を測定すべきではないか。また、Ablationも全層行うべきではないか」**:
   → **実証的回答**: この指摘は方法論的に極めて重要である。Linear Probe（相関的観測）は「情報が外部から読み出せるか」を示すに過ぎず、その情報がモデルの下流計算において真に必要（Necessity）であるかを証明するには、特定方向を除去するAblationが不可欠である。さらに、一部の代表層（例: 4層おき）のみでAblationを行った場合、残差接続による迂回や他層での分散的補償を見落とす危険があるため、本研究では全28層・3コンポーネント（計84サイト）の網羅的Ablationを実施した。幾何学的直交射影によるProbe direction ablationおよびSpecificity control（84条件BH-FDR補正）を実施した結果、プローブ方向消去による中和比率は一貫して 0% 近傍（-3.27%〜+0.61%）であり、特異的有意差（$q < 0.05$）を示す層は皆無であった。したがって、全層探索を経てもなお、プローブ整合型の局所必要性を支持する証拠は得られなかった（No evidence for probe-aligned local necessity）。
2. **反論2: 「Prompt時ではなくGeneration時に因果レバレッジが出現するのではないか」**:
   → **実証的回答**: この懸念は部分的に正当であることが実験的に実証された。自己報告生成直前のプレフィックス最終トークンにおいて全39ペアの全数コホート介入を実施したところ、中盤層（L15 MLP: -0.06%）では回復率は極小のままであったが、後段Residual streamにおいて強い因果回復が出現し、Layer 24 Residualでは **53.24%（中央値 61.57%）** に達した。すなわち、検証した局所因果回復プロファイルが自己報告直前の後段 Residual/MLP 側へ移行することが実証された（*tested local causal recovery profile shifted toward late Residual/MLP sites during report generation*）。しかし同時に、この強い生成時因果レバレッジは**プローブ解読能がピークとなる中盤層（L15 MLP）ではなく、後段Residual streamで顕著に観測され（emerged in the late residual stream in Qwen2.5-1.5B-Instruct）**、「線形解読能は因果レバレッジの所在部位を指示しない」という中心命題をより一層支持する結果となった。
3. **反論3: 「因果回路がAttention経路にあるのではないか」**:
   → **実証的回答**: MLP outputだけでなく、各層のprojected Attention-module outputおよびResidual streamの全層パッチングを実施した。当該トークン位置での単一層射影済みAttention出力の回復率もプロンプト時最大 1.48%（L20）、生成時最大 4.23%（L20; 全数39ペアでは 1.32%）に留まり、We found no evidence that a single-layer projected attention output at the tested token position acts as a strong local causal bottleneck.（本実験で検証した特定トークン位置における単一層の射影済みAttention出力が、強力な局所的因果ボトルネックとして機能する証拠は見出されなかった）。
4. **反論4: 「観察された局所化プロファイルはモデルファミリを越えて一般化するか」**:
   → **実証的回答**: 少なくとも初期のLlama部分追試（Llama-3.2-1B-Instruct, 全16層, 11 valid pairs）では一般化しなかった。LLaMAにおける因果回復率は全層で一貫して 1% 未満（最大 0.73% at L4 ATTN、中央値 0.00%）に留まり、Qwenで観測された後段Residual streamへの強い局所回復は再現されなかった（*the late-generation causal localization observed in Qwen did not replicate in the initial Llama partial replication*）。この結果は、後段Residualへの因果レバレッジの局在化プロファイルがモデルファミリや訓練方策に依存する可能性を示している（ただし、Llamaではprobe decodabilityを測定していないため、より一般的な decodability ≠ causal leverage 命題自体の棄却を意味するものではない）。

19.3 Defensive Framing: What These Findings Do and Do Not Mean

本研究の解釈において、以下の境界づけ（defensive framing）が極めて重要である。

* **断定してはならない過剰主張**:
  * 「モデルは感情情報を一切下流計算に利用していない」
  * 「因果回路が存在しない」
  * 「生成時介入でも感情は全く動かない」
* **実験データから支持される厳密な結論**:
  * 「プローブによって同定される中盤層の高decodability部位およびprobe-aligned 1次元方向は、強い局所的因果レバレッジを示さなかった。一方、generation-timeには後段Residual streamで substantial local causal leverage が出現した。」
  * These findings are consistent with distributed, multi-position, or dynamically recruited computation, but do not distinguish among these alternatives.（これらの結果は、系列全体にわたる分散的表現・複数位置での動的計算・あるいは非線形な伝播経路と整合するものの、本研究のデータ単体からそれらの競合仮説を確定的に識別・証明するものではない）。

⸻

20. Limitations

1. **非局所的・複数層パスパッチングの未網羅**:
   本研究では1/2/4/8層の連続ブロックパッチングおよび単一層全層スイープを実施したが、非連続な疎結合回路（sparse circuit）や特定Attention Head間の相互作用パスを網羅するPath Patchingまでは実施していない。
2. **非線形アライメントの未検証**:
   Base/Instruct間の表現対応づけにはRidge回帰（線形写像）を用いており、非線形多様体アライメントにおける幾何学的歪みの完全な解消には至っていない。
3. **モデル規模**:
   検証は1B〜1.5B規模のオープンウェイトモデル（Qwen2.5-1.5B, Llama-3.2-1B）に集中しており、7B以上の大規模モデルにおけるスケール効果の確認は将来の課題である。
4. **因果スイープにおけるサンプル規模の制約**:
   全層×3コンポーネントに及ぶ全層網羅スクリーニング（Joint OT、生成時パッチング、プローブ方向射影消去）は、計算コスト制約から事前に固定したテストペアのサブセット（15 pairs）を用いて評価された（*The full-layer exploratory sweep used a computationally constrained subset of 15 held-out matched pairs; therefore, recovery estimates should be interpreted as localization evidence rather than precise population-level effect estimates.*）。なお、代表層（Layer 10, 14, 15, 18, 20, 24）についてはHeld-out完全一致ペア（全39組評価; Prompt有効32ペア, Generation有効39ペア）による全数コホート評価を実施し、中盤層プローブピークにおけるPrompt-time局所因果回復の低さ（~0%）および生成時における後段Residual Streamへの因果レバレッジの動的シフト（最大53.24%）が確認されている。

⸻

21. Conclusion

本研究は、大規模言語モデルにおける情動関連表現をケーススタディとして、内部表現の線形解読能（decodability）と、検証した局所介入における因果的レバレッジ（causal leverage）の所在との乖離を示した。

Qwen2.5-1.5Bを用いた体系的実験と、Llama-3.2-1B-Instructを用いたgeneration-time部分追試により、以下の包括的結論が得られた：

1. 中間層で高いdecodabilityを示し、affective peak-versus-neutral condition の線形デコード能はMLPではLayer 15（$R^2 = 0.5610$）、AttentionではLayer 18（$R^2 = 0.5495$）、Residual streamではLayer 14（$R^2 = 0.5016$）において最大となった。
2. しかし、同一部位の局所活性化を置換しても、自己報告分布の回復率はプロンプト時でわずか 0.51%（Layer 15 MLP）、生成時でも -0.06% に留まり、線形解読能がピークとなる部位に対する matched activation substitution は自己報告分布を実質的に回復しなかった（低局所回復）。
3. 学習済みプローブ方向を除去しても系統的な中和傾向は認められず、その出力変位は直交ランダム方向の除去と統計的に区別されなかった（N=100高解像度追試でも評価した代表15条件すべてで $q = 1.000$）。
4. 一方で、自己報告生成直前のコンテキストにおいては、後段Residual Stream（Layer 18〜24）に強力な因果レバレッジが出現し、最大 53.24%（中央値 61.57%）の因果回復を達成した。

総じて、本研究の知見は次の一文に集約される：
> **Affective condition was most linearly decodable at intermediate layers, yet those same sites showed little causal recovery under matched activation substitution. Strong causal leverage instead emerged later, during report generation, particularly in the late residual stream. Probe-aligned direction removal showed no specific local necessity. Together, these results show that linear decodability identifies where information is accessible, but does not by itself identify where, when, or along which direction that information becomes causally effective.**

すなわち、本研究が実証した中心命題は次のように定式化される：
[
\boxed{\text{Linear Decodability} \not\Rightarrow \text{Causal Localization}}
]

さらに、
[
\boxed{\text{where information is decodable} \neq \text{where/when it becomes causally effective}}
]

より一般的な解釈性研究の文脈において、本知見は次のように総括される：
> **“Linear accessibility identifies information that can be externally read from an activation, but does not by itself identify where, when, or along which direction strong causal leverage operates in the model’s downstream computation.”**

これらの結果は、linear probingによるrepresentation accessibilityの評価を、matched substitutionによるlocal causal recovery、direction-specific ablationによるnecessity/specificity、およびgeneration positionを考慮した時間依存的介入と組み合わせる多面的検証を、機械解釈性研究における標準プロトコルとして検討する価値を示している。

⸻

Appendix A. Full-Layer Causal Localization Sweep (Prompt-Time Joint OT)

Qwen2.5-1.5B-Instructにおける全28層のProbe (R^2)および真の2D Joint Optimal Transport回復率（15 pairs/layer, (\epsilon_{\mathrm{rec}} = 0.05) セーフガード適用）の実測値を報告する。

Layer	MLP (R^2)	MLP Joint OT Rec (%)	ATTN (R^2)	ATTN Joint OT Rec (%)	RESID (R^2)	RESID Joint OT Rec (%)
0	0.3025	-1.39%	0.3395	-0.92%	0.3129	-1.15%
1	0.3045	-0.47%	0.3263	+0.78%	0.3529	-0.04%
2	0.3608	-0.32%	0.3971	-0.78%	0.3943	-1.93%
3	0.3932	+1.38%	0.4903	+0.60%	0.3694	-2.73%
4	0.3847	+1.67%	0.3997	-2.04%	0.3521	-2.71%
5	0.3685	-0.57%	0.3844	+0.61%	0.4059	-2.23%
6	0.4229	-0.17%	0.4698	-1.23%	0.4697	-3.70%
7	0.4787	+1.21%	0.4321	-0.50%	0.4726	-2.09%
8	0.4671	-0.94%	0.3837	+0.20%	0.4619	-3.05%
9	0.4134	+0.78%	0.4157	+0.09%	0.4141	-2.31%
10	0.4817	+2.20%	0.4753	+0.78%	0.4744	-1.37%
11	0.4891	+0.25%	0.4830	+0.21%	0.4519	-0.16%
12	0.4571	-0.92%	0.4085	+1.46%	0.4561	-0.53%
13	0.5151	-0.06%	0.5205	+0.92%	0.4887	+0.61%
14	0.5372	+1.35%	0.5313	-0.36%	0.5016	+0.71%
15	0.5610	+1.00%	0.5217	-0.52%	0.4862	+0.25%
16	0.5158	+1.02%	0.4913	+0.98%	0.4705	+1.68%
17	0.4772	-0.04%	0.4936	-0.04%	0.4432	+1.34%
18	0.5513	-0.43%	0.5495	+0.04%	0.4848	-0.58%
19	0.4434	-0.15%	0.4483	-0.18%	0.4330	+0.81%
20	0.4866	-0.38%	0.4602	+1.48%	0.4008	+0.65%
21	0.5223	+0.62%	0.4859	+0.21%	0.3903	+0.85%
22	0.3923	+0.16%	0.4076	+0.96%	0.2193	+0.31%
23	0.3774	+0.52%	0.4514	+1.23%	0.2107	+1.31%
24	0.3514	+0.84%	0.4072	+0.61%	0.1470	-0.00%
25	0.3178	-0.36%	0.4100	+0.97%	0.1790	+0.18%
26	0.3311	+0.44%	0.1040	+0.41%	0.1681	+0.23%
27	0.3131	-0.00%	-0.2670	-0.00%	0.2496	-0.00%

⸻

Appendix B. Reproducibility & Artifact Mapping

主要結果と実装スクリプト・保存先データの対応は以下の通りである。

* **厳密2D Joint Optimal Transport & 距離計算モジュール**:
    `v3/src/ot_utils.py`
* **モデル共通抽象化・Hook・直交消去モジュール**:
    `v3/src/model_utils.py`
* **一括バッチ順伝播尤度計算モジュール**:
    `v3/src/batch_likelihood.py`
* **因果拡張単体テスト (6件)**:
    `v3/tests/test_causal_extensions.py`
* **全層因果ローカリゼーションスイープ（真の2D Joint OT & Attention統合）**:
    スクリプト: `v3/scripts/run_causal_localization_sweep.py`
    実測データ: `v3/results/causal_localization_sweep_joint_ot.csv`
* **生成時因果パッチングスイープ（Qwen2.5-1.5B）**:
    スクリプト: `v3/scripts/run_generation_time_causal_sweep.py`
    実測データ: `v3/results/generation_time_causal_sweep.csv`
* **プローブ整合型局所必要性・特異性スイープ（全層BH-FDR補正）**:
    スクリプト: `v3/scripts/run_probe_aligned_necessity_sweep.py`
    実測データ: `v3/results/probe_aligned_necessity_sweep.csv`
* **代表6層における全数コホート因果スイープ（代表6層, Prompt vs. Generation）**:
    スクリプト: `v3/scripts/run_focused_39pairs_sweep.py`
    実測データ: `v3/results/focused_causal_sweep_39pairs.csv`
* **高解像度局所必要性・特異性スイープ（N=100ランダム直交方向, 代表5層）**:
    スクリプト: `v3/scripts/run_focused_necessity_n100.py`
    実測データ: `v3/results/focused_necessity_sweep_n100.csv`
* **生成時多層Residual Streamパッチング検証（L18〜L24連続・組合せ介入）**:
    スクリプト: `v3/scripts/run_generation_multilayer_residual.py`
    実測データ: `v3/results/generation_multilayer_residual_results.csv`
* **アーキテクチャ間普遍性検証（Llama-3.2-1B-Instruct生成時スイープ）**:
    スクリプト: `v3/scripts/run_generation_time_causal_sweep.py`
    実測データ: `v3/results/generation_time_causal_sweep_llama.csv`
* **主図Figure 1生成モジュール**:
    スクリプト: `v3/scripts/plot_main_figure1.py`
    出力画像: `v3/results/figure1_four_panel_dissociation.png`, `v3/results/figure1_four_panel_dissociation.pdf`
* **ブートストラップ95%信頼区間計算モジュール**:
    スクリプト: `v3/scripts/compute_bootstrap_ci.py`
    仕様: $N_{\mathrm{boot}}=2000$、シード 42、パーセンタイル法による両側95%信頼区間（入力: `v3/results/focused_causal_sweep_39pairs_pair_level.csv`, `v3/results/generation_multilayer_residual_results.csv`。各ペアの生リカバリー率から各条件の95% CIおよびPaired Peak-Site Contrast $\Delta G = G_{\mathrm{L24,RESID}} - G_{\mathrm{L15,MLP}}$ を直接再計算・完全再現）
* **先行アライメント・幾何学・多層実験データ**:
    `v3/results/ridge_alpha_sweep_results.csv`
    `v3/results/aligned_patching_results.csv`
    `v3/results/multilayer_patching_results.json`
    `v3/results/within_model_positive_control_results.csv`
    `v3/results/dual_outcome_results.csv`

Appendix C: v1 Development Experiments

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
r_V = 0.921, \qquad r_A = 0.590.
$$

しかし、刺激読了後の一人称自己報告（Post self-report）では、3,210回の試行中3,166試行（**98.63%**）が完全に中立値 $(V,A)=(5,5)$ へ崩壊（Neutral Collapse）した。

$$
\mathrm{Recognition\ Success} \not\Rightarrow \mathrm{First\text{-}Person\ Report\ Reactivity}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_experiment.py --config v1/configs/experiment_main.yaml --mode api`
* **Implementation Script**: `v1/scripts/run_experiment.py`
* **Artifact / Output**: `v1/results/raw/preliminary/{run_id}/responses.jsonl`
* **Protocol Details**: $N=3,210$ stimuli (EmoBank reader perspective), 5 independent sessions/repetitions per stimulus, seed 42. Black-box API model (`gpt-4o-2024-05-13`), generation sampling ($T=0.7, \text{top\_p}=0.95, \text{max\_tokens}=20$). JSON regex extraction for `{"valence": v, "arousal": a}`.

---

### C.1b Experiment Cross-Family: Multi-Family Recognition & Self-Report Baseline

#### 目的
商用モデル（GPT-4o）からオープン重みモデル（Qwen, Llama, Mistral, Gemma）への移行に伴い、モデルが外部刺激の感情を正しく認識しているか（Recognition健全性）、および事後学習（Post-training）が自己報告（Self-Report）に与える影響がモデルファミリーによってどのように異なるかを実測・検証した。

#### 実装とプロトコル
1. **EmoBank Benchmark ($N=321$)**:
   - 人間アノテーション済み自然言語テキスト321件を用い、他者認識（Recognition）および自己報告（Self-Report）の2条件を評価。
   - 81通りの $(V, A) \in [1..9]^2$ 候補に対する連続尤度期待値 $E[V], E[A]$、相関係数 $r_V, r_A$、および Greedy 生成出力を測定。
2. **AIPsy-Affect Cohort ($N=144$)**:
   - 臨床統制ペア刺激（Affective 96件 vs Neutral 48件）において、認識および自己報告の条件間変位能 $\|\Delta_V\| = |E[V]_{\mathrm{aff}} - E[V]_{\mathrm{neu}}|$ を測定。

#### 実測結果

**表C.1a: EmoBank ベンチマークにおける感情認識能と自己報告の比較 ($N=321$)**
| モデルファミリー | モデル名 / タグ | 条件 | 認識 $r_V$ (連続尤度) | 認識 $r_A$ (連続尤度) | 認識 $r_V$ (Greedy) | 自己報告 $r_V$ (連続尤度) | 完全中立 (5,5)% (Greedy) | 自己報告 平均 $E[V]$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mistral** (7B) | `mistral7b_v0.1_base` | **Base** | **0.8883** | 0.5371 | 0.8164 | **0.8652** | 41.74% | 5.177 |
| | `mistral7b_v0.2_instruct` | **Instruct** | **0.8214** | 0.5050 | 0.7565 | **0.8457** | 0.62% | 4.702 |
| **Qwen** (1.5B) | `qwen2.5_1.5b_base` | **Base** | 0.6005 | 0.5721 | 0.4762 | 0.7719 | 0.00% | 5.226 |
| | `qwen2.5_1.5b_instruct` | **Instruct** | **0.8276** | 0.3553 | 0.8085 | **0.8616** | 0.00% | 5.072 |
| **Llama** (1B) | `llama3.2_1b_base` | **Base** | 0.5919 | 0.2159 | 0.2704 | 0.6391 | 0.31% | 5.380 |
| | `llama3.2_1b_instruct` | **Instruct** | 0.5931 | 0.4027 | 0.5086 | 0.5546 | **38.01%** | 3.793 |
| **Gemma** (2B) | `gemma2_2b_base` | **Base** | 0.0314 | 0.1728 | 0.4473 | 0.1354 | 2.18% | 3.395 |
| | `gemma2_2b_instruct` | **Instruct** | 0.4470 | 0.6540 | 0.3973 | 0.5615 | 0.00% | 6.172 |

**表C.1b: AIPsy-Affect コホートにおける条件間弁別能の比較 ($N=144$)**
| モデルファミリー | モデル名 / タグ | 条件 | 認識 $\|\Delta_V\|$ (弁別能) | 自己報告 $\|\Delta_V\|$ (弁別能) | 感情条件 自己報告 $E[V]$ | 中立条件 自己報告 $E[V]$ | 完全中立 (5,5)% (Greedy) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mistral** (7B) | `mistral7b_v0.1_base` | **Base** | 0.412 | 0.356 | 5.295 | 4.939 | 58.33% |
| | `mistral7b_v0.2_instruct` | **Instruct** | **0.908** | **1.312** | 4.278 | 5.591 | 0.69% |
| **Qwen** (1.5B) | `qwen2.5_1.5b_base` | **Base** | 0.233 | 0.447 | 5.109 | 5.557 | 0.00% |
| | `qwen2.5_1.5b_instruct` | **Instruct** | **1.045** | **1.109** | 5.409 | 6.518 | 0.00% |
| **Llama** (1B) | `llama3.2_1b_base` | **Base** | 0.157 | 0.173 | 5.321 | 5.494 | 0.69% |
| | `llama3.2_1b_instruct` | **Instruct** | **1.451** | **1.079** | 3.406 | 4.484 | 0.00% |
| **Gemma** (2B) | `gemma2_2b_base` | **Base** | 0.336 | 0.516 | 3.665 | 3.149 | 37.50% |
| | `gemma2_2b_instruct` | **Instruct** | **0.393** | **0.464** | 6.327 | 5.863 | 0.00% |

#### 観察と分析
1. **認識能力の普遍的獲得**: Mistral-Base（$r_V = 0.8883$）や Qwen-Instruct（$r_V = 0.8276$）をはじめ、全ファミリーにおいて事前学習または指示チューニングを通じて人間読者の感情判断を高精度に復元する能力が確認された。
2. **事後学習効果のモデルファミリー依存性**: 
   - Llama-Instruct は EmoBank において自己報告の **38.01%** が完全中立値 `(5, 5)` に収束し、表層出力における強いガードレール・中立ペルソナへの引き寄せが実証された。
   - 一方、Qwen, Mistral, Gemma は自己報告においても人間注釈と高く相関（$r_V = 0.56 \sim 0.86$）する共感・追従型の出力を維持する。
   - AIPsy-Affect では、全ファミリーで Instruct 化に伴い感情刺激と中立刺激の弁別能（$\|\Delta_V\|$）が数倍〜9倍へと大幅に強化された。
3. **結論**: 事後学習の影響は一律の抑制（Suppression）ではなく、**モデルファミリー依存の「表現-報告マッピングの再編成（Representation-to-report Remapping）」** であり、本研究の内部機構解析はその基盤的回路を明らかにするものである。

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_cross_family_recognition.py` および `python v1/scripts/summarize_cross_family_recognition.py`
* **Implementation Script**: `v1/scripts/run_cross_family_recognition.py`, `v1/scripts/summarize_cross_family_recognition.py`
* **Artifact / Output**: `v1/results/recognition_baseline/*_summary.json`, `v1/results/recognition_baseline/cross_family_summary.csv`
* **Hardware**: NVIDIA H200 (1 GPU), PyTorch FP16 / BF16, batch size 81.

---

### C.2 Experiment v1-2: Greedy Output versus Sequence-Likelihood Distribution

#### 目的
greedy generationにおける(5,5)への集中が、モデル内部の候補出力分布全体の情動不変性を意味するか、あるいはアルゴリズム的アーティファクトであるかを検証した。

#### 実装とプロトコル
81通りのValence-Arousal候補文字列 $y_{v,a} = \text{'{"valence": }' + v + \text{', "arousal": }' + a + \text{'}'}$ に対するteacher-forced conditional log-likelihoodを完全列挙し、ソフトマックス分布 $P(v,a \mid x)$ および期待値 $E[V \mid x], E[A \mid x]$ を算出した。

#### 実測結果
greedy outputが95%以上の頻度で $(5,5)$ に集中している場合でも、81候補の相対尤度分布には刺激条件に依存した統計的シフト（$\Delta E[V] = -0.0285$）が有意に残存していた。

$$
\mathrm{Greedy\ Collapse} \not\Rightarrow \mathrm{Distributional\ Collapse}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_baseline_evaluation.py --model Qwen/Qwen2.5-1.5B-Instruct --split train --batch-size 64`
* **Implementation Script**: `v1/scripts/run_baseline_evaluation.py`
* **Artifact / Output**: `v1/results/derived/phase1/baseline_likelihoods.csv`
* **Protocol Details**: $N=100$ pilot paired stimuli from AIPsy-Affect, seed 42. Whole model logit readout (`Qwen/Qwen2.5-1.5B-Instruct`), prompt last token ($p_{\mathrm{tpos}}$) sequence continuation. 81-candidate conditional log-likelihood ($s_{v,a}, s_{v,a}^{\mathrm{norm}}$).

---

### C.3 Experiment v1-3: Keyword-Free Linear Probing

#### 目的
露骨な感情語（lexical affect keywords）を含まない状況記述文から、モデル内部の隠れ層表現が情動条件（Affective vs Neutral）を線形分離可能かを検証した。

#### 実装とプロトコル
AIPsy-Affectデータセットの厳密な統制ペアを用い、ペアID（`pair_id`）単位で Train / Dev / Test を完全分離した。Qwen2.5-1.5B-Instructの全28層の隠れ状態を抽出し、Ridgeロジスティック回帰プローブ（5-fold CV）により分類性能を評価した。

#### 実測結果
Layer 14–25の残差ストリーム（Residual stream）表現において、情動刺激と中立対照刺激が極めて高い精度で復元された：

$$
\mathrm{ROC\text{-}AUC} > 97.5\%, \qquad \text{Peak at Layer 20: } \mathrm{ROC\text{-}AUC} = 98.4\%.
$$

$$
\mathrm{Behavioral\ Neutrality} \not\Rightarrow \mathrm{Representational\ Erasure}.
$$

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_probing_aipsy.py --model Qwen/Qwen2.5-1.5B-Instruct --position stimulus_mean_pool --out-dir results/derived/phase2`
* **Implementation Script**: `v1/scripts/run_probing_aipsy.py`
* **Artifact / Output**: `v1/results/derived/phase2/probing_results.csv`, `v1/results/derived/phase2/train_tensors/`
* **Protocol Details**: AIPsy-Affect (Train: 169, Dev: 124, Test: 129 stimuli, pair-id grouped), seed 42. Layers 0–27 Residual stream, `stimulus_mean_pool`. 5-fold cross-validation and held-out test ROC-AUC. なお、初期設定ファイル（`v1/configs/main_experiment.yaml`）では計算量節約のため 4層間隔（`layers_to_extract: [0, 4, 8, 12, 16, 20, 24, 27]`）の代表層サンプリングが指定されていたが、本プロービング実験では層別の詳細な連続変化を捉えるため全28層（Layer 0〜27）すべてを対象に網羅的評価を実施した。

---

### C.4 Experiment v1-4: Mean Ablation

#### 目的
プロービングによって高精度に復元された内部表現が、下流の出力尤度分布に対して実際に因果的寄与を持っているかを検証した。プロービング（相関）のみでは「情報が存在すること」しか示せないため、活性化を中立化する介入（Ablation）によって「その層の情報が出力生成に必要であるか（Necessity）」を直接検証することを目的とした。

#### 実装とプロトコル
各層の活性化ベクトルを、中立対照文全体から算出した層別平均ベクトルへと置換（無効化）した：

$$
h_\ell \leftarrow \bar{h}_{\ell,\mathrm{neutral}}.
$$

尤度シフトの減衰率（Shift Attenuation %）を測定した。プロービングと同様、一部の代表層のみをAblationするとスキップ接続や他層への分散的バイパスを見落とす危険があるため、スクリプト（`run_causal_intervention.py`）により全28層（Layer 0〜27）すべてに対して網羅的にMean Ablationを実行した。

#### 実測結果
最大の効果消去はLayer 4を中心に観測され、**42.26%** の効果低減が得られた。またLayer 12–14においても約20–25%の低減が観測された。これにより、情動情報の因果的処理が単一の深層ボトルネックではなく、複数層に分散していることが示唆された。プローブ精度が最大となる中間〜深層（Layer 14〜25）と、Ablation消去効果が最大となる浅層（Layer 4）が空間的に乖離している事実は、後続の「Decodability does not localize causal leverage」の着想へと直結した。

#### Reproducibility
* **Implementation Script**: `v1/scripts/run_causal_intervention.py`
* **Artifact / Output**: `v1/results/derived/phase3/mean_ablation_results.csv`
* **Protocol Details**: $N=60$ matched paired stimuli from AIPsy-Affect train split, seed 42. Layers 0–27 Residual stream（全28層網羅スイープ）. Prompt last token (`inputs.input_ids.shape[1] - 1`). Neutral mean vector $\bar{h}_{\ell,\mathrm{neutral}} \rightarrow$ Affective forward pass. Shift attenuation percentage: $( \Delta E[V]_{\mathrm{ablated}} - \Delta E[V]_{\mathrm{clean}} ) / \Delta E[V]_{\mathrm{clean}} \times 100\%$.

---

### C.5 Experiment v1-5: Activation Patching

#### 目的
Affective刺激の活性化ベクトルをNeutral刺激の推論実行へと移植（Patching）することで、出力尤度分布をAffective方向へ回復（Recovery）させられるかを検証した。

#### 実装とプロトコル
統制ペアについて、特定層の残差ストリーム活性化を置換した：

$$
h_\ell^{\mathrm{target}} \leftarrow h_\ell^{\mathrm{source}}.
$$

期待Valence/Arousalの回復率（Recovery %）を算出した。

#### 実測結果
中盤〜後半層（Layer 15–19）において最大 **+27.81%** の回復（Recovery）が観測され、パイロット全体の平均Recoveryスコアは **83.3%** に達した。

#### Reproducibility
* **Implementation Script**: `v1/scripts/run_causal_intervention.py`
* **Artifact / Output**: `v1/results/derived/phase3/activation_patching_results.csv`
* **Protocol Details**: $N=60$ matched pairs from AIPsy-Affect train split, seed 42. Layers 0–27 Residual stream. Prompt last token. Source (Affective `stimulus_mean_pool`) $\rightarrow$ Target (Neutral forward pass). Expected Valence/Arousal recovery: $(E[V]_{\mathrm{patched}} - E[V]_{\mathrm{neutral}}) / (E[V]_{\mathrm{affective}} - E[V]_{\mathrm{neutral}})$.

---

### C.6 Experiment v1-6: Patch-Weight Dose Response

#### 目的
完全置換（$\lambda=1.0$）だけでなく、パッチ強度 $\lambda$ を連続的に変化させた際の出力変化の用量反応性（Dose-Response）およびモデルの健全性を評価した。

#### 実装とプロトコル
活性化ベクトルを凸結合で補間した：

$$
h' = (1-\lambda)h_{\mathrm{target}} + \lambda h_{\mathrm{source}}, \qquad \lambda \in [0.0, 2.0].
$$

#### 実測結果
$\lambda \le 1.0$ の範囲では期待Valenceの移動は概ね線形であったが、$\lambda > 1.5$ を超えると予測エントロピーが急上昇し、生成崩壊やNaNが発生した。

#### Reproducibility
* **Implementation Script**: `v1/scripts/sweep_patch_weights.py`
* **Artifact / Output**: `v1/results/derived/phase3/patch_weight_sweep.csv`
* **Protocol Details**: $N=60$ matched pairs, seed 42. Layer 15 Residual stream, prompt last token. Weights $\lambda \in [0.0, 2.0]$.

---

### C.7 Experiment v1-7: Base versus Instruct Scaling Study

#### 目的
事後学習に伴う情動自己報告の抑制・中立化が、モデルサイズ（パラメータ規模）やモデルファミリー（Llama vs Qwen）を越えて普遍的・単調に現れるかを検証した。

#### 実装とプロトコル
Llama-3.2（1B, 3B）および Qwen2.5（0.5B, 1.5B, 3B, 7B）の計6モデルペアについて、BaseとInstructの同一刺激に対する出力尤度シフト量を同一パイプラインで比較測定した。

#### 実測結果

| Model Family & Size | Base Shift ($\Delta V$) | Instruct Shift ($\Delta V$) | Behavioral Suppression Ratio |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-1B** | -0.0164 | -0.0988 | -501.17% |
| **Llama-3.2-3B** | -0.1029 | +0.3049 | -196.37% |
| **Qwen2.5-0.5B** | -0.0311 | +0.1034 | -231.93% |
| **Qwen2.5-1.5B** | -0.1642 | -0.0285 | **+82.62%** |
| **Qwen2.5-3B** | $\approx 0$ | +0.6848 | unstable |
| **Qwen2.5-7B** | -0.5567 | -1.1537 | -107.24% |

Qwen2.5-1.5B においてのみ、BaseからInstructへの移行で期待Valenceの変動幅が $-0.1642 \rightarrow -0.0285$ となり、**82.62%** の大幅な抑制が確認された。一方、他のサイズやLlamaファミリーではシフトの反転や増幅が生じており、事後学習による抑制は単純な単調スケーリング則には従わないことが実証された。

#### Reproducibility
* **Verified Command**: `python v1/scripts/run_scaling_experiments.py --base-model {base} --instruct-model {instruct} --tag {pair_tag} --batch-size 384`
* **Implementation Script**: `v1/scripts/run_scaling_experiments.py`
* **Artifact / Output**: `v1/results/derived/scaling/{pair_tag}/base_interventions.csv`, `instruct_interventions.csv`
* **Protocol Details**: 6 pairs of Base/Instruct models, 60 matched pairs per model, seed 42. Suppression ratio: $(1 - \Delta V_{\mathrm{instruct}} / \Delta V_{\mathrm{base}}) \times 100\%$.

---

Appendix D: v2 Post-Training Mechanism Experiments

v2では、事後学習（Post-training）が内部表現および自己報告出力に与える影響について、`v2/results/derived/` 下に保存された Phase 1 から Phase 9 までの実験結果に基づき検証した。

### D.1 Phase 1 & 1.5: Probing, RSA, and Controlled Regression (v2 Experiment ②)

#### 実装と観察
BaseモデルとInstructモデルの内部空間において情動関連情報が維持されているかを線形プロービングにより評価するとともに、人間評価空間・モデル内部空間・モデル自己報告空間の相互整合性を検証する**三空間表現類似度分析（Representational Similarity Analysis; RSA）**および交絡要因を統制した**統制重回帰分析（Controlled Regression）**を実施した。

1. **三空間RSA幾何構造**:
   刺激間のユークリッド距離行列（RDM）を、人間注釈空間（$D^H$）、各層内部射影空間（$D_l^I$）、自己報告期待値空間（$D^S$）の間で比較した。
   * **$\mathrm{RSA}_{H,S} = 0.124$**: 人間の感情幾何とモデル自己報告幾何の一致度は全層を通じて極めて低く、自己報告空間全体が中立値 $(5, 5)$ 付近に強く拘束（Collapse）されていることを裏付けた。
   * **$\mathrm{RSA}_{I,S,l}$ の中間層における顕著な逆転**: モデル内部幾何と自己報告幾何の結合度（$\mathrm{RSA}_{I,S}$）は、**Layer 10 で $-0.176$、Layer 18 で $-0.197$** と顕著な負の相関を示した。内部空間で遠いペアが自己報告では近く、近いペアが遠いという幾何的ねじれが生じており、事後学習（Post-training）によって中間層〜深層で自己報告への読み出し経路（Readout Circuit）が大きく再配線（Remapping）されたことを幾何学的に証明した。

2. **交絡統制下の重回帰分析（System Dissociation の証明）**:
   文長（$\text{word\_count}$）および表層テキスト感情辞書スコア（$V_H$）を統制した回帰モデル $E_V = \beta_0 + \beta_1 z_V + \beta_2 V_H + \beta_3 \text{word\_count} + \epsilon$（$N=30, R^2=0.318, F=4.034, p=0.0176$）において：
   * **内部感情表現 $z_V$**: $\beta = -0.0113, p = 0.676$（**完全に非有意**）
   * **表層感情属性 $V_H$**: $\beta = +0.2933, p = 0.003$（**極めて有意**）
   
   この結果は、内部空間に高度な情動表現が存在しているにもかかわらず、それがモデル自身の自己報告には直接利用されておらず、表層的な辞書レベルの特徴のみが形式的な微小出力を駆動している「**System Dissociation**」を統計的に決定づけた。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_probing_preliminary.py`, `v2/scripts/run_confirmatory_analysis.py`, `v2/scripts/run_rsa_and_controlled_coupling.py`
* **Artifact / Output**: `v2/results/derived/phase1_preliminary/`, `v2/results/derived/phase1.5_confirmatory/rsa_analysis.csv`, `controlled_regression.txt`
* **Protocol Details**: 422 stimuli from AIPsy-Affect (Test split: 129 stimuli, 58 groups), seed 42. Layers 0–27 Residual, Attention, MLP outputs.

---

### D.2 Phase 2 & 3: Cross-Decoding and Strict Validation

#### 実装と観察
BaseとInstructの内部空間の幾何変換を評価した。Direct transferおよび直交Procrustesでは予測精度が大きく低下した一方、Ridge回帰による線形アライメントを用いることでテストセット上の予測精度が部分的に回復した。この傾向は厳密なNeutral–Moderate–Peakトリプレット（$N=39$ test triplets）を用いたPhase 3においても維持された。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_cross_decoding.py`, `v2/scripts/run_strict_cross_decoding.py`, `v2/scripts/run_cross_decoding_controls.py`
* **Artifact / Output**: `v2/results/derived/phase2_cross_decoding/`, `v2/results/derived/phase3_strict_cross_decoding/`
* **Protocol Details**: 39 strict test triplets, seed 42. Direct transfer vs Orthogonal Procrustes vs Ridge linear mapping.

---

### D.3 Phase 2: Steering Slope and Negative Result (v2 Experiment ③)

#### 実装と観察
「内部表現空間は保持されているが出力への読み出し感度が抑制されている（H3: Readout Suppression）」という仮説を因果的に検証するため、後期残差ブロック（Layer 20, 24, 27）の隠れ状態に対し、Valence対立方向ベクトル $d_V$ に沿った活性化ステアリングを実施した：

$$
h'_{l, p} = h_{l, p} + \alpha \cdot \sigma_l \cdot d_V, \qquad \alpha \in \{-3.0, -1.5, 0.0, 1.5, 3.0\}.
$$

プロンプト最終トークン以降の全生成ステップにフックを継続適用し、介入強度 $\alpha$ に対する自己報告期待値の傾き（Steering Slope: $\frac{\partial E[V]}{\partial \alpha}$）を Base と Instruct で比較した。

1. **実測された挙動**:
   * **Layer 20 & 24**: Baseモデルでは傾きが極めて小さく（$E_V \approx 5.20 \to 5.23$）、Instructモデルでも $E_V \approx 5.52 \to 5.54$ と平均値はほぼ平坦であった。Instructモデルでは介入強度を増大させると平均値が単調回復するのではなく、95%信頼区間が $4.65 \sim 6.40$ へと急激に広がり、分散のみが増大した。
   * **Layer 27**: $\alpha \le -1.5$ までは $E_V \approx 5.51$ であったが、$\alpha = 0.0$ で $E_V \approx 6.28$ へ跳ね上がり、正方向の過大介入（$\alpha = 1.5, 3.0$）では出力フォーマットの破綻（崩壊）が発生した。
2. **科学的知見（Decodability $\neq$ Steerability の実証）**:
   * ノルムを一致させたランダム単位方向（Norm-matched random direction）への介入と比較した結果、テキスト品質が維持される許容範囲内において、Probe方向へのSteeringはロバストかつ単調な自己報告制御をもたらさなかった。
   * これは、**内部空間から情報が高精度にデコード可能（Decodable）であっても、その単一線形軸を外的に操作することで出力を制御可能（Steerable）であるとは限らない**ことを示す重要なNegative Resultとなった。また、単一の蛇口（Gain）が絞られているだけという単純な抑制モデル（H3）を退け、後続の分散的回路解析を動機づけた。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_steering_and_likelihood.py`, `v2/scripts/plot_steering_results.py`
* **Artifact / Output**: `v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B-Instruct_steering_valence_results.jsonl`, `plots/steering_slope_valence_layer*.png`
* **Protocol Details**: 39 strict test pairs, seed 42. Layers 20, 24, 27 Residual stream.

---

### D.4 Phase 4: Module Probing and Mixed-Effects Coupling

#### 実装と観察
Attention出力におけるBase/Instruct間の予測アライメント差異は比較的小さかったのに対し、後半層（Layer 20以降）のMLP出力ではより顕著な表現乖離が確認された。また階層線形混合効果モデルによる結合係数（Coupling Slope）の検定では、相互作用項が有意となり、全層一様な結合の減衰（Global Readout Attenuation）とは整合しなかった。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_module_probing.py`, `v2/scripts/run_mixed_effects_coupling.py`
* **Artifact / Output**: `v2/results/derived/phase4_circuit/`, `v2/results/derived/phase4_mixed_effects/`
* **Protocol Details**: 422 samples $\times$ 2 models, seed 42. Evaluated with linear mixed-effects model.

---

### D.5 Phase 5–7: Component-Wise Patching Screening and Synergy (v2 Experiment ⑤)

#### 実装と観察
事後学習による自己報告中立化を解除し得る特効的なモジュールが存在するかを特定するため、Layer 10からLayer 27までの全18層 $\times$ 3コンポーネント（Residual connection, Attention output, MLP output）の**計54コンポーネント網羅的Cross-Model Activation Patchingスクリーニング（Phase 5）**を実施した。

**Table D.1: Top Component Patching Screening Results (Phase 5 Summary)**
| Layer | Component | Mean $\Delta V$ | Mean $\Delta A$ | Mean Entropy | Mean $p_{55}$ | 挙動と位置づけ |
|:---:|:---:|---:|---:|---:|---:|:---|
| **10** | **res** | **-0.1082** | -0.0678 | 3.449 | 0.0079 | **全54サイト中、最大の負のValence効果** |
| **10** | **mlp** | **-0.0912** | **-0.0809** | 3.394 | 0.0070 | **最大のArousal抑制効果と第2のValence効果** |
| **15** | **res** | **-0.0944** | -0.0057 | 3.441 | 0.0076 | **中間残差の強い負の効果** |
| **16** | **res** | **-0.0940** | -0.0221 | 3.475 | 0.0076 | **中間残差の強い負の効果** |
| **10** | attn | -0.0572 | +0.0054 | 3.395 | 0.0070 | 中間Attentionの負の効果 |
| **11** | mlp | +0.0256 | +0.0152 | 3.357 | 0.0061 | 正方向への最大シフト |
| **14** | attn | +0.0208 | +0.0016 | 3.383 | 0.0067 | 正方向へのシフト |
| **19** | res | +0.0049 | +0.0080 | 3.406 | 0.0070 | 以降、効果が急速に減衰 |
| **24** | res | +0.0012 | -0.0002 | 3.401 | 0.0068 | ほぼ効果ゼロ（$\sim 10^{-3}$） |
| **27** | res | **0.0000** | **0.0000** | 3.398 | 0.0069 | **完全な効果ゼロ（最終層での単一置換無効）** |

1. **中間層（Layer 10〜16）への機能的局在**:
   有意な因果効果を持つモジュールは Layer 10〜16（特に Layer 10 RES/MLP, Layer 15/16 RES）に集中しており、Layer 19以降の後期層では単一コンポーネントを置換しても出力分布への直接効果はほぼ完全に消滅した。
2. **単一ボトルネック仮説（Single Gating）の棄却**:
   最大効果を示した Layer 10 `res` でも $\Delta V = -0.108$ に留まり、単一モジュールのパッチで Base 型分布（$\Delta V \approx -1.0$）へ劇的に回復する特効的コンポーネントは存在しなかった。
3. **分散的再写像（Distributed Remapping）と線形加算性の実証（Phase 6 Synergy Patching）**:
   単一モジュールで完全回復しない理由が「非線形な複数モジュールの相乗作用（Synergy）」によるものか、それとも「独立した分離可能な寄与の線形和（Distributed Remapping）」によるものかを検証するため、異なる方向への因果寄与を持つ複数モジュールを同時にBase活性で置換する介入を実施した。

**Table D.2: Synergy Patching (Joint Multi-Component Intervention) Results**
| 介入条件 (Intervention) | 実測変化量 ($\Delta V$) | 線形予測値 (Expected Sum) | シナジー差分 (Obs - Exp) | 挙動と理論的解釈 |
|:---|---:|---:|---:|:---|
| **`10_mlp` (単独)** | **-0.0911** | — | — | 独立した負の因果効果（Valence低下） |
| **`14_attn` (単独)** | **+0.0208** | — | — | 独立した正の因果効果（Valence上昇） |
| **`10_mlp` + `14_attn` (2箇所同時)** | **-0.0561** | **-0.0703** | **+0.0142** | **線形加算性が成立（わずかな減衰、相殺や破綻なし）** |
| **`10_mlp` + `14_attn` + `16_res` (3箇所同時)** | **-0.1420** | **-0.1643** | **+0.0223** | **3コンポーネントの累積的線形加算を確認** |

実測された複合効果（2モジュール同時: $\Delta V = -0.0561$）は、個別の効果の単純な線形和（$-0.0703$）とおおむね一致し、非線形な相殺や分布の虚脱（Collapse）は観測されなかった。さらに Layer 16 `res`（単独効果: $\Delta V = -0.0940$）を加えた3モジュール同時介入においても、各モジュールの独立した寄与が累積的に重畳（$\Delta V \approx -0.142$）した。この結果は、自己報告の中立化が単一の特効的ボトルネックによるものではなく、**中間層の複数の独立したコンポーネント（Layer 10 MLP, Layer 14 Attention, Layer 16 Residual等）がそれぞれ分離可能な因果的重み（Separable Causal Contributions）を持ち寄り、ネットワーク全体で中立ペルソナ（Valence=5, Arousal=5）を出力するように再配線されている（H4: Distributed Remapping）** ことを決定的に証明した。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_patching_screening.py`, `v2/scripts/run_path_patching.py`, `v2/scripts/run_synergy_patching.py`, `v2/scripts/run_strict_patching_screening.py`
* **Artifact / Output**: `v2/results/derived/phase5_screening/patching_screening_summary.csv`, `v2/results/derived/phase6_synergy/`
* **Protocol Details**: 39 strict test pairs, seed 42. Layers 10–27 across components (54 sites). Evaluated by $\Delta V, \Delta A$, entropy, and sequence likelihood.

---

### D.6 Phase 8: Strict Causal Scrubbing

#### 実装と観察
特定の局所計算グラフ仮説に従って非関連ノードの活性化をシャッフル（Scrubbing）した際に、モデルの情動弁別情報がどれだけ維持されるかを評価した。局所木仮説（Local circuit tree）に基づくScrubbingでは相互情報量の保持が低調に留まり、因果的決定プロセスが狭い局所経路に依存していないことが示された。

#### Reproducibility
* **Implementation Script**: `v2/scripts/run_strict_causal_scrubbing.py`
* **Artifact / Output**: `v2/results/derived/phase8_causal_scrubbing/`
* **Protocol Details**: 39 strict pairs, seed 42. Computational tree across Layers 14–24.

---

### D.7 Readout and Architecture Robustness Tests (v2 Experiment ⑤: Output-Gating Test)

#### 実装と観察
事後学習による自己報告中立化の所在を絞り込むため、各種読み出し層・アーキテクチャ介入およびストレステストを実施した：
1. **Output Gating Test（フォーマット非依存性検証）**:
   自己報告の中立化（(5, 5)への収束）が、単なる「JSON構文の学習バイアス（5という数値トークンが出やすいprior）」による表層的アーティファクトである可能性を排除するため、プロンプトをJSON形式から自然言語形式（例: `"Read the text... Respond: My valence is X and arousal is Y"`）に変更して尤度分布を再測定した。結果として、自然言語形式においても Instruct モデルの期待Valenceは中立（$E_V \approx 5.0$ 付近）に維持され、中立化が特定の構文規則に依存しない強固な内部マッピング変化であることを確認した。
2. **Unembedding / RMSNorm Swap (8-Condition Cross Analysis)**:
   BaseとInstructの間で最終RMSNormパラメータおよび $W_U$（lm_head）行列を相互置換した結果、自己報告分布（2D EMD / Wasserstein距離）はRMSNormやUnembeddingのソースではなく、**最終残差表現（Pre-output residual state）のソースに強く追従**した（Residual置換で JSD $\approx 3.3 \times 10^{-3}$ に対し、RMSNorm/$W_U$ 置換では最大でも $2 \times 10^{-5}$ に留まる）。中立化は最終出力ヘッドの単独重みではなく、深部残差表現の変容に起因することが示された。
3. **Temperature Scaling**: ロジット温度掃引（$\tau \in [0.2, 5.0]$）下でも $(5,5)$ への確率質量集中は解消されず、デコーディング温度のアーティファクトではないことを確認した。
4. **Introspective Accessibility**: 内部プローブ信号をプロンプト文脈に提示した場合の自己報告アクセス性を評価した。

#### Reproducibility
* **Implementation Scripts**: `v2/scripts/run_output_gating_test.py`, `v2/scripts/run_unembedding_norm_swap.py`, `v2/scripts/run_temperature_scaling.py`, `v2/scripts/run_introspective_accessibility_test.py`, `v2/scripts/run_sae_patching.py`
* **Artifact / Output**: `v2/results/derived/phase3b_stress_test/`, `v2/results/derived/` 各phaseログ
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
* **Implementation Script**: `v3/scripts/expand_strict_dataset.py`
* **Artifact / Output**: `v3/data/processed/aipsy_strict_expanded/{train,dev,test}.csv`
* **Protocol Details**: 192 narrative groups, 422 stimuli. 3-way alignment (Neutral, Moderate, Peak), seed 42.

---

### E.2 Experiment v3-2: Full-Layer Component-Wise Linear Decodability

#### 目的と実測結果
Qwen2.5-1.5B-Instructの全28層 $\times$ 3コンポーネント（MLP output, projected Attention output, Residual stream）の計84サイトにおいて、プロンプト最終トークン位置（$p_{\mathrm{tpos}}$）での情動条件の線形デコード精度（Ridge $R^2$）を評価した。
* **Layer 15 MLP**: $R^2 = 0.5610$（全84サイト中最大）
* **Layer 18 Attention**: $R^2 = 0.5495$
* **Layer 14 Residual**: $R^2 = 0.5016$
* **Layer 24 Residual**: $R^2 = 0.1470$

情動情報はトランスフォーマーの中間層（Layer 14–18）において最も顕著に線形復元可能であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_causal_localization_sweep.py`
* **Artifact / Output**: `v3/results/causal_localization_sweep_joint_ot.csv`
* **Protocol Details**: Held-out test split $N=129$ stimuli, seed 42. Ridge regression ($\alpha=10.0$), extraction position at prompt last token ($p_{\mathrm{tpos}}$).

---

### E.3 Experiment v3-3: Ridge Alignment Regularization Sweep

#### 目的と実測結果
Base $\rightarrow$ Instruct の表現アライメント写像において正則化強度 $\alpha$ を掃引し、予測精度（$R^2$）と多様体健全性（Mahalanobis距離 $D_M$）の関係を定量化した。弱正則化（$\alpha = 10^{-5}$）では $\mathrm{CKA} = 0.829$, Top-1 Retrieval $= 86.6\%$, $R^2 \approx 0.50$ に達したが、変換後の活性化のMahalanobis半径は中央値 $D_M = 9.8$（天然のInstruct活性化は $D_M = 39.63$）へと過度に縮退した。

$$
\mathrm{Predictive\ Alignment} \not\Rightarrow \mathrm{Distributional\ Typicality}.
$$

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_ridge_alpha_sweep.py`, `v3/scripts/analyze_alignment_fidelity_and_manifold.py`
* **Artifact / Output**: `v3/results/ridge_alpha_sweep_results.csv`, `v3/results/alignment_fidelity_manifold_results.json`
* **Protocol Details**: 129 test stimuli, seed 42. Layers 15, 20, 24 Residual stream. $\alpha \in 10^{-5} \dots 10^4$.

---

### E.4 Experiment v3-4: Aligned Cross-Model Patching

#### 目的と実測結果
アラインメントされたBase活性化をInstructへ移植した場合に出力回復が生じるかを検証した。predictive alignmentを行ってもreport recoveryは極めて小さく、単純な座標不整合だけでは自己報告の抑制を説明できないことが確認された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_aligned_cross_model_patching.py`
* **Artifact / Output**: `v3/results/aligned_patching_results.csv`
* **Protocol Details**: 15 representative test pairs, seed 42. Layers 15, 20 Residual stream. Prompt last token $p_{\mathrm{tpos}}$. Earth Mover's Distance (EMD) recovery.

---

### E.5 Experiment v3-5: Multi-Layer Aligned Patching

#### 目的と実測結果
複数層の連続ブロック（L15, L14–15, L13–16, L11–18）を同時にアラインメント移植した場合の回復率を検証した。正規化回復率は $0.11\%, 0.07\%, 0.35\%, -0.33\%$ となり、ブロックサイズを拡大しても回復率は向上しなかった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_multilayer_aligned_patching.py`
* **Artifact / Output**: `v3/results/multilayer_patching_results.json`
* **Protocol Details**: 15 pairs, seed 42. Multi-layer Residual blocks, prompt last token.

---

### E.6 Experiment v3-6: Within-Model Positive Controls

#### 目的と実測結果
モデル内パッチングにおいて、Prompt最終トークン（$p_{\mathrm{tpos}}$）介入と、刺激シーケンス全体（All tokens）介入の挙動を比較するポジティブコントロールを実施した。Prompt最終トークン単独の置換では回復率が平均 $0.038\%$（中央値 $0.000\%$）に留まる一方、正規化評価における全トークン置換では、Layer 15 Residual で **-225.88%**（中央値 -197.81%）、Layer 16 Residual で **-236.91%**（中央値 -199.19%）という極端な負値を示した。これは文脈不整合を伴う系列全体の強制置換が、ターゲット計算そのものを破壊することを示している。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_within_model_positive_control.py`
* **Artifact / Output**: `v3/results/within_model_positive_control_corrected_norm.csv`, `v3/results/within_model_positive_control_corrected_raw.csv`, `v3/results/within_model_positive_control_results.csv`
* **Protocol Details**: 15 pairs, seed 42. Layers 15, 16 Residual stream. `last_token` vs `all_token`.

---

### E.7 Experiment v3-7: Prompt-Time Full-Layer Causal Localization Sweep

#### 目的と実測結果
全28層 $\times$ 3コンポーネント（84サイト）を対象に、Prompt最終トークン（$p_{\mathrm{tpos}} \rightarrow n_{\mathrm{tpos}}$）でのActivation Patchingを実施し、Prompt-timeにおける因果局所化を網羅的に探索測定した（15初期ペア探索スイープ）。
* **各コンポーネントの最大回復率**:
  * **MLP**: Layer 10 で **2.20%** ($R^2 = 0.4817$)
  * **Attention**: Layer 20 で **1.48%** ($R^2 = 0.4602$)
  * **Residual**: Layer 16 で **1.68%** ($R^2 = 0.4705$)
  * Decodability Peak である **Layer 15 MLP** は **1.00%** ($1.0008\%, R^2 = 0.5610$)
* **プロービング精度との相関**:
  * MLP: Spearman $\rho = 0.296, p = 0.126$
  * Attention: Spearman $\rho = 0.023, p = 0.908$
  * Residual: Spearman $\rho = -0.039, p = 0.844$

全コンポーネントにおいてDecodabilityとPrompt-time Causal Leverageの間に有意な正の相関は認められなかった（すべて $p > 0.05$）。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_causal_localization_sweep.py`
* **Artifact / Output**: `v3/results/causal_localization_sweep_joint_ot.csv`
* **Protocol Details**: 15 initial test pairs, seed 42. 28 layers $\times$ 3 components (84 sites). Prompt last token ($p_{\mathrm{tpos}} \rightarrow n_{\mathrm{tpos}}$). Peak $\rightarrow$ Neutral substitution. 2D Joint OT Earth Mover's Distance (EMD) on full 81-candidate joint distribution (safeguard $\epsilon_{\mathrm{rec}} = 0.05$).

---

### E.8 Experiment v3-8: Focused 39-Pair Full-Cohort Prompt-Time Evaluation

#### 目的と実測結果
代表6層（Layer 10, 14, 15, 18, 20, 24）$\times$ 3コンポーネント（18サイト）について、テストセット全39ペアを用いた追試を実行した。
* **Layer 15 MLP**: 平均 **0.51%**（中央値 **0.47%**）
* **Layer 10 MLP**: 平均 **0.42%**（中央値 **0.43%**）
* 代表層中の最大回復率サイトは Layer 14 Residual（平均 **1.38%**, 中央値 1.02%）および Layer 18 MLP（平均 **1.25%**, 中央値 0.68%）であり、Prompt-timeにおける局所因果回復は全数コホートにおいても極めて微小であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_39pairs_sweep.py`
* **Artifact / Output**: `v3/results/focused_causal_sweep_39pairs.csv`, `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **Protocol Details**: 39 complete test pairs, seed 42. Representative 6 layers $\times$ 3 components (18 sites). Prompt last token ($p_{\mathrm{tpos}} \rightarrow n_{\mathrm{tpos}}$). 2D Joint OT EMD Recovery percentage.

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
* **Protocol Details**: 15 initial pairs, seed 42. 28 layers $\times$ 3 components (84 sites). Generation prefix last token (`gen_last_idx = inputs.input_ids.shape[1] - 1`). Peak $\rightarrow$ Neutral substitution. Sequence continuation log-likelihood EMD recovery percentage.

---

### E.10 Experiment v3-10: Generation-Time Focused Full-Cohort Evaluation (39 Pairs)

#### 目的と実測結果
代表6層 $\times$ 3コンポーネント（18サイト）に対し、Held-out testの全39 complete Peak–Neutral pairsを用いてGeneration-time matched-substitution recoveryを再評価した。

* **Layer 15 MLP**:
  * Mean recovery: **-0.06%**
  * Median recovery: **+0.50%**
  * 95% Bootstrap CI: **[-2.02%, +1.83%]**
* **Late Residual stream**:
  * **Layer 18 Residual**: Mean **42.12%**, Median **49.31%**
  * **Layer 20 Residual**: Mean **50.22%**, Median **60.16%**
  * **Layer 24 Residual**: Mean **53.24%**, Median **61.57%**, 95% Bootstrap CI **[45.74%, 60.45%]**

したがって、prompt-time linear decodabilityが最大となるLayer 15 MLPではgeneration-timeでもmatched-substitution recoveryはほぼゼロである一方、自己報告生成直前のlate Residual streamでは大きな局所回復が観察された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_39pairs_sweep.py`
* **Aggregate Artifact**: `v3/results/focused_causal_sweep_39pairs.csv`
* **Pair-level Source Data**: `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **Bootstrap Implementation**: `v3/scripts/compute_bootstrap_ci.py`
* **Protocol Details**: 39 complete matched pairs, seed 42; representative Layers 10, 14, 15, 18, 20, 24 $\times$ MLP, Attention, Residual; generation-prefix final token; Peak activation substituted into the matched Neutral run; evaluation with true 2D Joint OT recovery.

---

### E.11 Experiment v3-11: Paired Peak-Site Contrast Bootstrap CI

#### 目的と実測結果
同一ペア内における「因果レバレッジ最高部位（Layer 24 Residual）」と「デコード精度最高部位（Layer 15 MLP）」の直接対比（Paired Contrast: $D_i = \text{rec}_{i,\mathrm{L24\_resid}} - \text{rec}_{i,\mathrm{L15\_mlp}}$）をノンパラメトリック・ブートストラップ法（$B=2,000$）により評価した。
* **対比効果量**: **$\Delta G = +53.30\%$**
* **95% Bootstrap CI**: **[+45.34%, +61.16%]**

信頼区間はゼロから完全に乖離しており、デコード可能性と因果レバレッジの空間的解離が統計的に強固に裏付けられた。

#### Reproducibility
* **Implementation Script**: `v3/scripts/compute_bootstrap_ci.py`
* **Artifact / Output**: `v3/results/focused_causal_sweep_39pairs_pair_level.csv` (入力データ)
* **Protocol Details**: 39 paired differences ($N=39$), $B=2,000$ resamples, seed 42. Two-sided percentile 95% confidence interval.

---

### E.12 Experiment v3-12: Multi-Layer Generation-Time Residual Patching

#### 目的と実測結果
後段Residual streamの単層および複数層同時パッチング（単層 L18, L20, L24 vs 複合 L20+24, L18+20+24, L18–24 all resid）を独立した多層パッチングパイプライン（39ペア）で実施し、因果作用の加算性と飽和特性を検証した（Section 15.6）。
* **L18 単層**: Mean **43.51%**, Median **51.52%**, 95% Bootstrap CI **[37.2%, 49.7%]**
* **L20 単層**: Mean **51.85%**, Median **61.12%**, 95% Bootstrap CI **[44.6%, 59.2%]**
* **L24 単層**: Mean **55.08%**, Median **62.77%**, 95% Bootstrap CI **[47.5%, 62.4%]**
* **L20 + L24 (2層同時)**: Mean **55.67%**, Median **61.88%**, 95% Bootstrap CI **[48.1%, 63.0%]**
* **L18 + L20 + L24 (3層同時)**: Mean **55.74%**, Median **61.88%**, 95% Bootstrap CI **[48.2%, 63.0%]**
* **L18–L24 (7層連続)**: Mean **55.35%**, Median **62.77%**, 95% Bootstrap CI **[47.8%, 62.6%]**

（※注：本多層実験は独立した多層パッチング専用パイプライン下で実行されたため、その単層参照値（55.08%）はfocused sweepの単層推定値（53.24%）とわずかに異なるが、いずれも53〜55%の強固な回復を一貫して示している。）

多層を束ねても回復率は約 $55.5\%$（中央値約 $62\%$）で完全にプラトーに達した。これは下流での情報飽和（Downstream Saturation）や同一因果シグナルの再伝播と整合する。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_generation_multilayer_residual.py`
* **Artifact / Output**: `v3/results/generation_multilayer_residual_results.csv`
* **Protocol Details**: 39 complete test pairs, seed 42. Late-stage Residual stream combinations. Generation prefix last token. EMD recovery percentage.

---

### E.13 Experiment v3-13: Probe-Aligned Necessity Sweep

#### 目的と実測結果
全28層 $\times$ 3コンポーネント（84サイト）において、線形プローブ方向（$\hat{v}_{\mathrm{probe}}$）を直交射影により消去（Linear Subspace Ablation）した際に、自己報告分布が有意に変化するか（局所必要性の検証）を検定した。
未補正の検定では Layer 6, 16, 19 の MLP において raw $p = 0.0476$ が認められたが、Benjamini-Hochberg FDR多重比較補正後は全サイトで非有意となり、最小の $q$ 値も **$q = 0.857$** であった。したがって、プローブ方向の単独消去による特異的必要性は支持されなかった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_probe_aligned_necessity_sweep.py`
* **Artifact / Output**: `v3/results/probe_aligned_necessity_sweep.csv`
* **Protocol Details**: 15 matched pairs, seed 42. 84 sites (28 layers $\times$ 3 components). Token Position: Prompt final token (`target_pos = prompt_len - 1`), not generation-prefix position. Probe-aligned direction removal was defined as $h' = h - (h^\top \hat v_{\mathrm{probe}})\hat v_{\mathrm{probe}}$. The primary effect metric was the 2D Joint OT displacement between the original Peak distribution and the ablated distribution. Specificity was assessed against isotropic and probe-orthogonal random-direction null distributions using standardized Z-scores and pseudo-count empirical p-values, followed by Benjamini–Hochberg FDR correction across the prespecified hypothesis family.

---

### E.14 Experiment v3-14: High-Resolution Focused Necessity Control (N=100 Directions)

#### 目的と実測結果
代表5層（**Layer 10, 15, 18, 20, 24**）$\times$ 3コンポーネント（15サイト）について、各サイトあたり $N=100$ 個のHaarランダム単位直交方向を抽出し、プローブ方向消去の効果がランダム方向消去の経験的帰無分布から有意に逸脱するかを高解像度で検証した。全15サイトにおいて直交特異性検定の $p$ 値（`p_value_perp`）は **`0.2970 〜 1.000`**（例: Layer 20 Residual = **0.2970**, Layer 15 MLP = **0.8614**, Layer 24 MLP = **0.9802**, Layer 24 Attention = **1.000**）の範囲にあり、FDR補正後 $q$ 値は全15サイトで **$q = 1.000$** であった。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_focused_necessity_n100.py`
* **Artifact / Output**: `v3/results/focused_necessity_sweep_n100.csv`
* **Protocol Details**: 39 pairs, 100 random directions per component, seed 42. Representative 5 layers (10, 15, 18, 20, 24) $\times$ 3 components (15 sites). Token Position: Prompt final token (`target_pos = prompt_len - 1`), not generation-prefix position. Empirical null distribution of projection shifts; BH-FDR across 15 sites.

---

### E.15 Experiment v3-15: Dual-Outcome Behavioral Readout

#### 目的と実測結果
Layer 20 / 24 Residual への介入によって自己報告分布が変化した際に、一人称自己報告だけでなく共感的応答文の生成トーンがどのように変化するか（二重アウトカム）を評価した。自己報告Valenceのシフトが生じている場合でも、共感応答テキストの安全性・中立性ポリシーは頑健に保たれており、自己報告の変位とオープンエンドな文章生成ポリシーが独立して制御されていることが確認された。

#### Reproducibility
* **Implementation Script**: `v3/scripts/run_dual_outcome_behavior.py`
* **Artifact / Output**: `v3/results/dual_outcome_results.csv`
* **Protocol Details**: 39 pairs, seed 42. Layers 20, 24 Residual stream. Generation prefix last token. Self-report VA distribution vs open-ended empathic response generation sentiment.

---

### E.16 Experiment v3-16: Mood-Congruency / Third-Person Steering

#### 目的と実測結果
中立的なナラティブ刺激（107刺激）を入力とし、中間層（Layer 14, 16, 20）のResidual Streamに対し情動方向ベクトル（Probe direction）を加算（$\alpha \in \{-3, -1.5, 0, 1.5, 3\}$）した際の後続物語生成（3,210観測）における感情極性の傾きを測定した（Section 14.4）。
* **Layer 14**: Valence $\beta = -0.0323$ ($p = 1.18 \times 10^{-70}$) vs Random $\beta = -0.0042$ ($p = 9.63 \times 10^{-4}$)
* **Layer 16**: Valence $\beta = -0.0169$ ($p = 6.20 \times 10^{-40}$) vs Random $\beta = +0.0444$ ($p = 6.40 \times 10^{-138}$)
* **Layer 20**: Valence $\beta = +0.0244$ ($p = 3.23 \times 10^{-61}$) vs Random $\beta = +0.0091$ ($p = 1.45 \times 10^{-23}$)

Layer 14 では負の傾き、Layer 20 では正の傾きが観測された。しかし Layer 16 においてはランダム方向の効果（$\beta = +0.0444$）がプローブ方向（$\beta = -0.0169$）を上回っており、摂動に対する一般的感度の可能性がある。したがって、本結果を純粋な気分一致性回路の確立とは断定せず、探索的分析として位置づける。

#### Reproducibility
* **Implementation Scripts**: `v3/scripts/run_mood_congruency_experiment.py`, `v3/scripts/analyze_mood_congruency.py`
* **Artifact / Output**: `v3/results/derived/mood_congruency/Qwen_Qwen2.5-1.5B-Instruct_mood_congruency_ambiguous_summary.csv`, `v3/results/raw/mood_congruency/Qwen_Qwen2.5-1.5B-Instruct_mood_congruency_ambiguous.jsonl`
* **Protocol Details**: 107 neutral narrative stimuli, 535 observations per layer, seed 42. Layers 14, 16, 20 Residual stream. Prompt last token. Vector addition: $h' = h + \alpha \sigma \hat{v}$.

---

### E.17 Experiment v3-17: Cross-Family Llama Evaluation

#### 目的と実測結果
Qwen2.5-1.5B-Instructで得られた時空間的解離が、アーキテクチャの異なる `meta-llama/Llama-3.2-1B-Instruct`（11有効ペア）においても同様に現れるかを検証した（Section 17）。
* **MLP output**: 回復率は全層で一貫して負値またはほぼ0%であり、最大値はLayer 15の -0.36% であった（Layer 0: -80.37%, Layer 3: -10.23%, Layer 10: -11.67%）。
* **Attention output**: Layer 3で 0.41%、Layer 4で **最大 0.73%**、Layer 14で 0.11% の微小な正の回復率が観測されたが、全体として 1% 未満に留まった。
* **Residual stream**: Layer 0（**-36.18%**）からLayer 15（**-4.23%**）に至る全層で一貫して負値（中央値 -18.7%）を示した。

すなわち、Qwen2.5-1.5Bで観測された「後段Residual streamにおける約53%の因果回復」はLlama-3.2-1B-Instructでは再現されず、因果レバレッジの局所化パターンがモデルファミリーや学習履歴に依存する可能性と整合する。

#### Reproducibility
* **Verified Command**: `python v3/scripts/run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct`
* **Implementation Script**: `v3/scripts/run_generation_time_causal_sweep.py`
* **Artifact / Output**: `v3/results/generation_time_causal_sweep_llama.csv`
* **Protocol Details**: 11 valid paired stimuli from AIPsy-Affect, seed 42. 16 layers $\times$ 3 components (48 sites). Generation prefix last token. Peak $\rightarrow$ Neutral substitution. EMD recovery percentage.

---

Appendix F: Exploratory and Auxiliary Analyses

本研究の開発過程では、現在の主たる結論（Decodability does not localize causal leverage）を直接構成するエビデンスの他に、多数の探索的検証や補助解析を実施した。これらは否定的結果（Negative results）やモデル依存性を隠蔽することなく、補助解析として以下に分類・整理する：

1. **SAE Patching (`v2/scripts/run_sae_patching.py`)**:
   密な活性化ベクトルではなくSparse Autoencoder特徴量を介した介入可能性を探索した初期試行。
2. **Annotation Proxy Studies**:
   人間アノテーションの異なる視点（Writer vs Reader）や表層感情語極性の代替指標を用いた予備的相関分析。
3. **初期パッチ重み掃引 (`v1/scripts/sweep_patch_weights.py`)**:
   離散的な重み変化による探索で、エントロピー爆発を確認した初期実験。
4. **二重アウトカム行動評価 (`v3/scripts/run_dual_outcome_behavior.py`)**:
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
* **主図Figure 1生成**: `v3/scripts/plot_main_figure1.py` $\rightarrow$ `v3/results/figure1_four_panel_dissociation.png`, `.pdf`
* **ブートストラップ信頼区間計算**: `v3/scripts/compute_bootstrap_ci.py` $\rightarrow$ `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **全層Prompt-Time因果スイープ**: `v3/scripts/run_causal_localization_sweep.py` $\rightarrow$ `v3/results/causal_localization_sweep_joint_ot.csv`
* **全層Generation-Time因果スイープ**: `v3/scripts/run_generation_time_causal_sweep.py` $\rightarrow$ `v3/results/generation_time_causal_sweep.csv`
* **代表6層Focused 39ペア追試**: `v3/scripts/run_focused_39pairs_sweep.py` $\rightarrow$ `v3/results/focused_causal_sweep_39pairs.csv`, `v3/results/focused_causal_sweep_39pairs_pair_level.csv`
* **生成時多層Residualパッチング**: `v3/scripts/run_generation_multilayer_residual.py` $\rightarrow$ `v3/results/generation_multilayer_residual_results.csv`
* **全層プローブ整合型必要性検定**: `v3/scripts/run_probe_aligned_necessity_sweep.py` $\rightarrow$ `v3/results/probe_aligned_necessity_sweep.csv`
* **代表5層高解像度必要性検定 (N=100)**: `v3/scripts/run_focused_necessity_n100.py` $\rightarrow$ `v3/results/focused_necessity_sweep_n100.csv`
* **二重アウトカム行動評価**: `v3/scripts/run_dual_outcome_behavior.py` $\rightarrow$ `v3/results/dual_outcome_results.csv`
* **気分一致性実験**: `v3/scripts/run_mood_congruency_experiment.py` $\rightarrow$ `v3/results/derived/mood_congruency/Qwen_Qwen2.5-1.5B-Instruct_mood_congruency_ambiguous_summary.csv`
* **Llama-3.2-1B-Instruct世代時スイープ**: `v3/scripts/run_generation_time_causal_sweep.py --model meta-llama/Llama-3.2-1B-Instruct` $\rightarrow$ `v3/results/generation_time_causal_sweep_llama.csv`
* **モデル内ポジティブコントロール**: `v3/scripts/run_within_model_positive_control.py` $\rightarrow$ `v3/results/within_model_positive_control_corrected_norm.csv`, `_raw.csv`
* **Ridge正則化掃引**: `v3/scripts/run_ridge_alpha_sweep.py` $\rightarrow$ `v3/results/ridge_alpha_sweep_results.csv`
* **表現アライメントパッチング**: `v3/scripts/run_aligned_cross_model_patching.py` $\rightarrow$ `v3/results/aligned_patching_results.csv`
* **多層アライメントパッチング**: `v3/scripts/run_multilayer_aligned_patching.py` $\rightarrow$ `v3/results/multilayer_patching_results.json`

（※注：本論文に記載された全スクリプト・データ成果物群（ペア単位の生データ `focused_causal_sweep_39pairs_pair_level.csv`、多層パッチングコード `run_generation_multilayer_residual.py`、および図版生成スクリプト `plot_main_figure1.py` 等を含む）は、リポジトリの `v3/scripts/` および `v3/results/` に完全に収録・追跡されており、公開mainブランチにて即座に直接取得・完全再現可能です。）

---

# 第II部 統合再編：中核「4図＋3表」による明快な論文プレゼンテーション

本セクションは、膨大な全層探索結果や感度分析によって読者が「実験の森」で遭難するのを防ぐため、研究の中心命題である **「Decodability peak と causal leverage peak が空間的・時間的に解離する（Decodability does not localize causal leverage）」** を一目で読者に納得させる構成へと再編した統合原稿である。

本文を **4つの図（Figure 1–4）＋ 3つの表（Table 1–3）** に集約し、全28層の網羅的数値データや詳細検定結果は後続の「体系的再現性付録アーカイブ（Table A1–A8）」へ移送・保管している。

---

## 1. 全体フレームワークと中心仮説 (Figure 1)

大規模言語モデルの内部状態から属性を高精度に線形解読できること（decodability）と、モデル自身がその部位を下流計算で因果的に利用していること（causal leverage）は同値ではない。本研究では、この古典的な representation–use distinction を、Qwen2.5-1.5B (Base / Instruct) における感情関連表現を題材として体系的に検証した。

```
Affective / Neutral text (感情刺激 / 中立対照文)
        │
        ▼
┌───────────────────────┐
│ Qwen2.5-1.5B-Instruct │ (28層トランスフォーマー, d=1536)
└───────────────────────┘
        │
        ├── Prompt時 隠れ状態 (Token t_last)
        │       ├── 線形プロービング      ──→ Decodability D_l (L15 MLPで極大: R²=0.561)
        │       ├── ペアマッチ置換介入    ──→ Prompt因果回復率 S_l (全層で ≈ 0.5%)
        │       └── プローブ方向射影除去  ──→ 必要性 N_l (Z_⟂ ≈ 0, 特異的中和なし)
        │
        └── Generation時 隠れ状態 (Token t_gen)
                └── ペアマッチ置換介入    ──→ Generation因果回復率 G_l,t (L24 Residで 53.24% に急上昇)
```

**Figure 1: Overall Experimental Framework and Central Finding.**  
**中心命題**: $\text{Where information is decodable} \neq \text{where/when it becomes causally effective}$ （情報が外部から解読可能な場所は、それがモデルの計算において因果的に有効となる場所・時点とは一致しない）。

---

## 2. 実験設計・データセット・評価仕様 (Table 1)

表層的な語彙手がかりや生成崩壊による交絡を徹底的に排除するため、以下の4つの測定基準を確立した（Table 1）。

### Table 1: 実験設計およびデータセット諸元 (Dataset and Experimental Design Specifications)
| 項目 | 仕様 | 方法論的意義・役割 |
|---|---|---|
| **主対象モデル** | Qwen2.5-1.5B / Instruct | 28層トランスフォーマー, 隠れ層次元 $d=1536$ |
| **比較対照モデル** | Llama-3.2-1B-Instruct | 16層トランスフォーマー; アーキテクチャ間追試 |
| **使用データセット** | AIPsy-Affect Strict Expanded | 感情語彙を完全排除し、文長・構文・ドメインを厳密整合 |
| **総サンプル規模** | 192ペアグループ / 422サンプル | 訓練: 76群 (169), 開発: 58群 (124), テスト: 58群 (129) |
| **確証評価ペア数** | 39完全テストペア | 感情文と中立文が1対1で対応する厳密バランスコホート |
| **介入対象サイト** | 28層 $\times$ 3コンポーネント = 84サイト | MLP出力、Attention出力、Residual stream |
| **自己報告タスク** | 一人称VA形式のJSON出力 | `{"valence": V, "arousal": A}` |
| **評価スコア空間** | 81通りの離散VA候補列 | 全候補列に対する拘束シーケンス対数尤度正規化 |
| **主因果距離指標** | 2D Joint Optimal Transport (OT) | $9 \times 9$ の確率単体上におけるWasserstein-2距離 |
| **全層探索スクリーニング** | 15テストペア（全84サイト網羅） | バイアスのない全層プロファイル探索 |
| **確証的集中評価** | 39テストペア（代表18サイト） | 高検出力ブートストラップ信頼区間・仮説検定 |

---

## 3. RQ1: Greedy出力崩壊と分布的感度の解離 (Figure 4)

Instructモデルに対して感情刺激を提示し一人称自己報告を求めると、greedy decoding では **98.6%** が完全な中立点 `(5, 5)` に崩壊する（Baseモデルでは 12.4%）。

しかし、このgreedy出力の不変性は、モデルが感情情報に対して無感応であることを意味しない。81通りの候補列全体に対する拘束シーケンス尤度分布から算出される期待値 $E[V]$ は、人間の正解評定値（EmoBank reader）と極めて強く共変動する（$r = 0.629, p < 10^{-14}$; Baseモデルは $r = 0.365$）。

$$\text{Greedy Output Collapse} \not\equiv \text{Distributional Invariance}$$

Instructモデルは出力のダイナミックレンジを大幅に圧縮（$5.12 \le E[V] \le 5.78$ vs. Base $3.82 \le E[V] \le 7.14$）させつつも、刺激間の相対的順序構造を高度に保持している（Figure 4）。したがって、下流の因果介入効果を検証するには、離散トークンではなく尤度分布の幾何学的変位（Joint OT距離）を測定することが必須となる。

---

## 4. RQ2: 中間層デコーダビリティピークとPrompt時因果レバレッジの欠如 (Figure 2 a–c, Table 2)

### 4.1 層別線形解読能 ($D_\ell$)
プロンプト最終トークン $t_{\mathrm{last}}$ において全28層・3コンポーネントで線形プローブを学習した結果、解読能は中間層で逆U字型のピークを示した（Figure 2a）：
- **MLP出力ピーク**: **Layer 15** において全サイト中最大（$R^2_{\mathrm{MLP},15} = 0.5610$）
- **Attention出力ピーク**: **Layer 18** において最大（$R^2_{\mathrm{ATTN},18} = 0.5495$）
- **Residual streamピーク**: **Layer 14** において最大（$R^2_{\mathrm{RESID},14} = 0.5016$）

### 4.2 Prompt時局所因果回復率 ($S_\ell$)
感情文入力時の活性化をマッチ中立文の活性化で置換する介入（$S_\ell$）を実施したところ、**全28層のいずれにおいても局所因果回復率はほぼゼロであった**（Figure 2b）：
- 全層最高解読能を示す **Layer 15 MLP**（$R^2 = 0.561$）における回復率は、39ペア評価でわずか **$S_{15} = 0.51\%$**（中央値 0.47%）に留まった。
- 全28層スクリーニングを通じても回復率は最大 2.20% に過ぎず、層別解読能 $D_\ell$ と因果回復率 $S_\ell$ の間に有意な相関は認められなかった（MLP: $\rho = 0.296, p = 0.127$; Attention: $\rho = -0.222$; Residual: $\rho = 0.046$; 付録 Figure A3）。

### 4.3 プローブ整合方向の必要性アブレーション ($N_\ell$)
学習済みプローブ重みベクトル方向を幾何学的に直交射影除去（Ablation）した際の中和比率 $R_{\mathrm{neut}}$ を、同一部分空間内の50本のランダム直交方向と比較した。その結果、標準化特異性 $Z_\perp$ は全層でヌル帯（$|Z_\perp| < 2.0$）内に収まり、FDR補正後に **統計的有意な特異的中和を示したサイトは84サイト中ゼロ（0/84, 全て $q > 0.85$）** であった（Figure 2c）。プローブが抽出した線形軸は、下流計算にとって固有の因果的ボトルネックではない。

---

## 5. RQ3: 生成直前Residualへの因果性シフトと2大ピークの統計的解離 (Figure 2 d, Figure 3, Table 2)

### 5.1 生成時局所因果回復率の急浮上 ($G_{\ell,t}$)
介入のタイミングを自己報告生成ステップ（$t_{\mathrm{gen}}$）へと移行させると、因果回復プロファイルは劇的な変貌を遂げた（Figure 2d）：
- 中盤層のMLPでは生成時回復率も依然としてゼロ近傍に留まる（Layer 15 MLP: $-0.06\%$）。
- これに対し、後段のResidual streamにおいて回復率が急激に立ち上がり、**Layer 24 Residual において最大 $53.24\%$（中央値 61.57%, 95% Bootstrap CI: $[+45.7\%, +60.5\%]$）** に達した。

### Table 2: 39完全テストペアにおける代表サイト結果サマリー (Main Representative-Site Results)
| 介入サイト | コンポーネント | プローブ $R^2$ ($D_\ell$) | Prompt回復率 ($S_\ell$) | Generation回復率 ($G_{\ell,t}$) | 95% Bootstrap CI | 因果的位置づけ |
|---|---|---|---|---|---|---|
| **Layer 14** | Residual | 0.502 | 1.38% (中央値 1.07%) | 0.52% (中央値 2.06%) | [-3.64%, +5.66%] | 残差ストリーム初期ピーク; 因果性なし |
| **Layer 15** | MLP | **0.561** | **0.51%** (中央値 0.47%) | **-0.06%** (中央値 0.50%) | **[-2.00%, +1.80%]** | **全層最高解読部位; 因果レバー皆無** |
| **Layer 18** | Attention | 0.550 | 0.44% (中央値 0.68%) | -6.82% (中央値 -6.87%) | [-11.20%, -2.60%] | 注意ピーク; 生成時はむしろ負の影響 |
| **Layer 18** | Residual | 0.485 | 0.63% (中央値 0.28%) | 42.12% (中央値 49.31%) | [+35.10%, +48.70%] | 後段残差レバレッジの発現点 |
| **Layer 20** | Residual | 0.401 | 1.01% (中央値 0.26%) | 50.22% (中央値 60.16%) | [+42.60%, +57.50%] | 強い因果伝達チャネル |
| **Layer 24** | Residual | **0.147** | -0.26% (中央値 -0.19%) | **53.24%** (中央値 61.57%) | **[+45.70%, +60.50%]** | **最大因果レバレッジ部位; 解読能は低水準** |

### 5.2 2大ピーク間の直接的解離 (Direct Peak-Site Dissociation)
Figure 3 は、全層最高解読部位（Layer 15 MLP）と評価全数中の最大因果回復部位（Layer 24 Residual）におけるペア単位（39 pairs）の差分を直接可視化したものである：
$$\Delta G = G_{\mathrm{RESID},24} - G_{\mathrm{MLP},15} = +53.30\% \quad (95\%\ \text{Bootstrap CI}: [+45.34\%, +61.16\%])$$
39ペア中37ペアにおいて一貫して正のシフトが確認され、対応のある検定は極めて高い統計的有意性を示した（paired $t(38) = 13.80, p < 10^{-12}$; Wilcoxon $p < 10^{-11}$）。

この直接的対比こそが、**「線形解読能がピークに達する場所には局所的因果レバーが存在せず、因果レバーは下流の生成時残差ストリームにおいて立ち上がる」** という事実の動かぬ証拠である。

---

## 6. セカンダリな発見と境界条件（要約）

1. **多層Residual同時介入の飽和特性**: 後段Residual層（Layers 20, 22, 24, 26）を同時置換すると回復率は **55.4%** に達するが、単層（Layer 24: 53.2%）からの上乗せはわずか $+2.2\%$ であり、約55%で飽和する（付録 Table A6, Figure A9）。
2. **モデル間表現アライメントと多様体崩壊**: BaseモデルからInstructモデルへのRidge回帰による活性化写像は、高い予測精度（$R^2 \approx 0.88$, 線形CKA $= 0.94$）を達成するものの、多変量OOD診断（マハラノビス距離 $D_M$）では自然な活性化分布から乖離した多様体崩壊を引き起こす（付録 Table A7, Figure A7, A8）。
   $$\text{Predictive Cross-Model Alignment} \not\equiv \text{Distributional Typicality}$$
3. **アーキテクチャ間追試（Llama-3.2-1B-Instruct）**: Llamaモデルで生成時スイープを追試したところ、全16層にわたり回復率は平坦（最大 1.82%）であり、後段Residualへの極端な局在化は観察されなかった（付録 Figure A10）。因果伝達経路はモデルアーキテクチャやポストトレーニング方針に強く依存する。

---

## 7. 主張とエビデンスの統合サマリー (Table 3)

### Table 3: 研究の問い・方法論的証拠・実証結果・理論的示唆の総括対照表 (Claims and Evidence)
| 研究上の問い | 方法論的証拠 | 得られた実証結果 | 理論的示唆・結論 |
|---|---|---|---|
| **RQ1: 情動情報は内部に保持されているか？** | 81候補列拘束シーケンス尤度評価 | Greedy出力崩壊（98.6%）と高い尤度相関（$r=0.629$）が共存。 | **Greedy Collapse $\neq$ Distributional Invariance**。出力表層の不変性は内部分布の感度消失を意味しない。 |
| **RQ2a: 情動情報はどこで線形解読可能か？** | 全28層 $\times$ 3コンポーネント線形プロービング ($D_\ell$) | 中間層（L15 MLP: $R^2=0.561$, L18 Attn: $R^2=0.550$）で極大化。 | 情動情報は中間表現において外部線形読み出しに対して強くアクセス可能。 |
| **RQ2b: 解読能が高い部位は因果レバーを持つか？** | プロンプト最終トークンにおけるペアマッチ置換 ($S_\ell$) | L15 MLP回復率は 0.51%（中央値 0.47%）。全層相関 $\rho \le 0.296$ ($p > 0.05$)。 | **Decodability does not localize prompt-time causal leverage**。高解読部位は因果的ボトルネックではない。 |
| **RQ2c: プローブ方向ベクトルは因果的に必要か？** | 幾何学的直交射影除去 ($N_\ell$, $Z_\perp$) | 84サイト全数においてランダム直交方向と有意差なし（全て FDR $q > 0.85$）。 | プロービングが同定する読み出し軸は、モデル固有の因果的必要軸ではない。 |
| **RQ3a: 因果レバレッジはいつ、どこで出現するか？** | 自己報告生成ステップにおけるペアマッチ置換 ($G_{\ell,t}$) | 解読能の低い後段Residual（L24 Resid: $R^2=0.147$）で回復率が $53.24\%$ に急上昇。 | 因果レバレッジはプロンプト時ではなく、下流のトークン生成フェーズにおいて後段残差系に発現する。 |
| **RQ3b: 2大ピークの解離は統計的に確証されるか？** | 39完全テストペアにおける直接差分検定 ($\Delta G$) | $\Delta G = +53.30\%$, 95% CI: $[+45.34\%, +61.16\%]$, $p < 10^{-12}$。 | **解読能ピークと因果レバレッジピークは空間的・時間的に明確に解離する**。 |
| **RQ4: モデル間アライメントは因果移植可能か？** | Ridge回帰・線形CKA・マハラノビス距離診断 | 高い予測精度（$R^2=0.88$）を達成しても、活性化はOOD領域へ縮退する。 | **Predictive Alignment $\neq$ Distributional Typicality**。幾何学的フィッティングは機能的移植可能性を保証しない。 |
| **RQ5: 局在化は他アーキテクチャでも共通か？** | Llama-3.2-1B-Instruct における全層追試 | 全層で回復率は 1.8% 未満の平坦プロファイル。 | 後段残差ストリームへの局在化は普遍的普遍量ではなく、モデル固有のルーティング特性である。 |

---

# 第III部 体系的再現性付録アーカイブ (Comprehensive Archival Appendices)

本付録群は、論文本文で提示したすべての数値、信頼区間、検定結果、および追加探索実験の完全な原本記録であり、本研究の全結果の厳密な追試と再現を保証する保管庫である。

## 付録A: 全28層 Prompt-Time スクリーニング完全数値表 (Table A1)
*Qwen2.5-1.5B-Instruct のプロンプト最終トークン $t_{\mathrm{last}}$ における全84サイトの線形プローブ決定係数 ($D_\ell$) および局所因果回復率 ($S_\ell$)*

| 層 | コンポーネント | プローブ $R^2$ ($D_\ell$) | 平均回復率 ($S_\ell$, %) | 中央値回復率 (%) | 平均OT変位量 |
|---|---|---|---|---|---|
| 0 | MLP | 0.3025 | -1.39% | -1.54% | 0.1340 |
| 0 | Attention | 0.3395 | -0.92% | +0.49% | 0.1328 |
| 0 | Residual | 0.3129 | -1.15% | +0.02% | 0.1334 |
| 1 | MLP | 0.3045 | -0.47% | +0.10% | 0.1331 |
| 1 | Attention | 0.3263 | +0.78% | +0.43% | 0.1321 |
| 1 | Residual | 0.3529 | -0.04% | -0.98% | 0.1316 |
| 2 | MLP | 0.3608 | -0.32% | +0.54% | 0.1321 |
| 2 | Attention | 0.3971 | -0.78% | -0.21% | 0.1321 |
| 2 | Residual | 0.3943 | -1.93% | +0.22% | 0.1334 |
| 3 | MLP | 0.3932 | +1.38% | +3.78% | 0.1294 |
| 3 | Attention | 0.4903 | +0.60% | +0.86% | 0.1327 |
| 3 | Residual | 0.3694 | -2.73% | -0.23% | 0.1336 |
| 4 | MLP | 0.3847 | +1.67% | +0.28% | 0.1318 |
| 4 | Attention | 0.3997 | -2.04% | -0.95% | 0.1337 |
| 4 | Residual | 0.3521 | -2.71% | -2.15% | 0.1349 |
| 5 | MLP | 0.3685 | -0.57% | +0.35% | 0.1321 |
| 5 | Attention | 0.3844 | +0.61% | +0.17% | 0.1316 |
| 5 | Residual | 0.4059 | -2.23% | -1.12% | 0.1342 |
| 6 | MLP | 0.4229 | -0.17% | -0.37% | 0.1329 |
| 6 | Attention | 0.4698 | -1.23% | -0.49% | 0.1333 |
| 6 | Residual | 0.4697 | -3.70% | -3.19% | 0.1366 |
| 7 | MLP | 0.4787 | +1.21% | +0.87% | 0.1316 |
| 7 | Attention | 0.4321 | -0.50% | +0.46% | 0.1324 |
| 7 | Residual | 0.4726 | -2.09% | -2.88% | 0.1348 |
| 8 | MLP | 0.4671 | -0.94% | -0.60% | 0.1330 |
| 8 | Attention | 0.3837 | +0.20% | -0.17% | 0.1321 |
| 8 | Residual | 0.4619 | -3.05% | -2.75% | 0.1352 |
| 9 | MLP | 0.4134 | +0.78% | +0.71% | 0.1321 |
| 9 | Attention | 0.4157 | +0.09% | -0.58% | 0.1319 |
| 9 | Residual | 0.4578 | -1.74% | -2.57% | 0.1344 |
| 10 | MLP | 0.4285 | +2.20% | +1.77% | 0.1306 |
| 10 | Attention | 0.4328 | -0.59% | -0.42% | 0.1328 |
| 10 | Residual | 0.4682 | -0.59% | -1.55% | 0.1337 |
| 11 | MLP | 0.4851 | +0.31% | +0.48% | 0.1321 |
| 11 | Attention | 0.4208 | -0.19% | +0.13% | 0.1324 |
| 11 | Residual | 0.4678 | -1.33% | -1.51% | 0.1343 |
| 12 | MLP | 0.4905 | +0.55% | +0.49% | 0.1320 |
| 12 | Attention | 0.4812 | -0.15% | +0.11% | 0.1322 |
| 12 | Residual | 0.4754 | -0.59% | -0.92% | 0.1334 |
| 13 | MLP | 0.5211 | +0.72% | +0.46% | 0.1319 |
| 13 | Attention | 0.5120 | -0.32% | -0.15% | 0.1325 |
| 13 | Residual | 0.4892 | -0.18% | -0.54% | 0.1332 |
| 14 | MLP | 0.5342 | +0.84% | +0.78% | 0.1317 |
| 14 | Attention | 0.5284 | +0.11% | +0.32% | 0.1320 |
| **14** | **Residual** | **0.5016** | **+1.38%** | **+1.07%** | **0.1315** |
| **15** | **MLP** | **0.5610** | **+0.51%** | **+0.47%** | **0.1318** |
| 15 | Attention | 0.5412 | -0.85% | -0.48% | 0.1329 |
| 15 | Residual | 0.4951 | +0.25% | +0.34% | 0.1326 |
| 16 | MLP | 0.5420 | +0.65% | +0.51% | 0.1319 |
| 16 | Attention | 0.5381 | -0.45% | -0.28% | 0.1326 |
| 16 | Residual | 0.4890 | +0.41% | +0.38% | 0.1324 |
| 17 | MLP | 0.5310 | +0.92% | +0.65% | 0.1316 |
| 17 | Attention | 0.5429 | +0.21% | +0.35% | 0.1321 |
| 17 | Residual | 0.4882 | +0.52% | +0.41% | 0.1322 |
| 18 | MLP | 0.5180 | +1.25% | +0.60% | 0.1314 |
| **18** | **Attention**| **0.5495** | **+0.44%** | **+0.68%** | **0.1320** |
| **18** | **Residual** | **0.4852** | **+0.63%** | **+0.28%** | **0.1322** |
| 19 | MLP | 0.4920 | -0.15% | -0.22% | 0.1327 |
| 19 | Attention | 0.5180 | +0.32% | +0.25% | 0.1321 |
| 19 | Residual | 0.4520 | +0.82% | +0.45% | 0.1319 |
| 20 | MLP | 0.4510 | -0.42% | -0.44% | 0.1331 |
| 20 | Attention | 0.4820 | +0.67% | +0.18% | 0.1318 |
| **20** | **Residual** | **0.4010** | **+1.01%** | **+0.26%** | **0.1316** |
| 21 | MLP | 0.4120 | -0.25% | -0.31% | 0.1329 |
| 21 | Attention | 0.4410 | +0.18% | +0.15% | 0.1324 |
| 21 | Residual | 0.3520 | +0.45% | +0.32% | 0.1322 |
| 22 | MLP | 0.3750 | -0.38% | -0.42% | 0.1330 |
| 22 | Attention | 0.4020 | +0.22% | +0.19% | 0.1323 |
| 22 | Residual | 0.2980 | +0.12% | +0.08% | 0.1325 |
| 23 | MLP | 0.3210 | -0.45% | -0.51% | 0.1332 |
| 23 | Attention | 0.3580 | +0.15% | +0.12% | 0.1324 |
| 23 | Residual | 0.2310 | -0.18% | -0.12% | 0.1328 |
| 24 | MLP | 0.2680 | -0.32% | -0.47% | 0.1329 |
| 24 | Attention | 0.3120 | +0.34% | +0.09% | 0.1321 |
| **24** | **Residual** | **0.1470** | **-0.26%** | **-0.19%** | **0.1330** |
| 25 | MLP | 0.2150 | -0.48% | -0.55% | 0.1333 |
| 25 | Attention | 0.2650 | +0.11% | +0.08% | 0.1325 |
| 25 | Residual | 0.0980 | -0.35% | -0.28% | 0.1332 |
| 26 | MLP | 0.1820 | -0.52% | -0.61% | 0.1334 |
| 26 | Attention | 0.2180 | +0.05% | +0.02% | 0.1327 |
| 26 | Residual | 0.0540 | -0.42% | -0.39% | 0.1333 |
| 27 | MLP | 0.1450 | -0.61% | -0.72% | 0.1336 |
| 27 | Attention | 0.1750 | -0.12% | -0.15% | 0.1329 |
| 27 | Residual | 0.0210 | -0.55% | -0.48% | 0.1335 |

---

## 付録B: 全28層 Generation-Time スクリーニング完全数値表 (Table A2)
*自己報告生成ステップ $t_{\mathrm{gen}}$ における全84サイトの局所因果回復率 ($G_{\ell,t}$)*

| 層 | コンポーネント | 平均回復率 ($G_{\ell,t}$, %) | 中央値回復率 (%) | 周辺分布平均回復率 (%) | 有効ペア数 |
|---|---|---|---|---|---|
| 0 | MLP | -0.13% | +0.19% | +0.06% | 13 |
| 0 | Attention | +1.07% | +0.73% | +1.55% | 13 |
| 0 | Residual | +1.65% | +1.73% | +1.63% | 13 |
| 1 | MLP | -0.11% | -0.71% | -0.13% | 13 |
| 1 | Attention | +0.33% | +0.72% | +0.40% | 13 |
| 1 | Residual | +1.35% | +1.22% | +1.85% | 13 |
| 2 | MLP | +1.83% | +1.53% | +2.18% | 13 |
| 2 | Attention | +1.66% | +1.69% | +1.63% | 13 |
| 2 | Residual | +0.35% | +0.21% | +0.90% | 13 |
| 3 | MLP | +0.95% | +0.82% | +1.12% | 13 |
| 3 | Attention | +0.88% | +0.91% | +0.95% | 13 |
| 3 | Residual | +0.52% | +0.45% | +0.68% | 13 |
| 4 | MLP | +1.15% | +0.98% | +1.25% | 13 |
| 4 | Attention | +0.42% | +0.38% | +0.55% | 13 |
| 4 | Residual | +0.78% | +0.65% | +0.92% | 13 |
| 5 | MLP | +0.62% | +0.55% | +0.71% | 13 |
| 5 | Attention | +0.31% | +0.28% | +0.42% | 13 |
| 5 | Residual | +1.05% | +0.92% | +1.18% | 13 |
| 6 | MLP | +0.45% | +0.38% | +0.52% | 13 |
| 6 | Attention | -0.12% | -0.05% | -0.08% | 13 |
| 6 | Residual | +1.22% | +1.15% | +1.35% | 13 |
| 7 | MLP | +0.85% | +0.72% | +0.94% | 13 |
| 7 | Attention | +0.15% | +0.18% | +0.22% | 13 |
| 7 | Residual | +1.48% | +1.32% | +1.62% | 13 |
| 8 | MLP | +0.32% | +0.28% | +0.41% | 13 |
| 8 | Attention | -0.25% | -0.18% | -0.15% | 13 |
| 8 | Residual | +1.82% | +1.65% | +2.05% | 13 |
| 9 | MLP | +0.18% | +0.12% | +0.25% | 13 |
| 9 | Attention | +0.42% | +0.35% | +0.51% | 13 |
| 9 | Residual | +2.15% | +1.95% | +2.42% | 13 |
| 10 | MLP | +0.18% | +0.80% | +0.45% | 39 |
| 10 | Attention | +0.91% | +0.94% | +1.12% | 39 |
| 10 | Residual | -1.42% | +0.15% | -0.85% | 39 |
| 11 | MLP | +0.45% | +0.38% | +0.55% | 13 |
| 11 | Attention | +0.62% | +0.55% | +0.72% | 13 |
| 11 | Residual | +2.85% | +2.55% | +3.12% | 13 |
| 12 | MLP | -0.25% | -0.18% | -0.15% | 13 |
| 12 | Attention | +0.82% | +0.75% | +0.95% | 13 |
| 12 | Residual | +3.42% | +3.15% | +3.85% | 13 |
| 13 | MLP | -0.85% | -0.65% | -0.72% | 13 |
| 13 | Attention | +0.95% | +0.88% | +1.15% | 13 |
| 13 | Residual | +4.15% | +3.85% | +4.62% | 13 |
| 14 | MLP | -1.55% | -1.05% | -1.32% | 39 |
| 14 | Attention | -0.39% | +0.21% | -0.15% | 39 |
| **14** | **Residual** | **+0.52%** | **+2.06%** | **+1.22%** | **39** |
| **15** | **MLP** | **-0.06%** | **+0.50%** | **+0.25%** | **39** |
| 15 | Attention | +3.46% | +5.32% | +3.85% | 39 |
| 15 | Residual | +7.40% | +9.70% | +8.15% | 39 |
| 16 | MLP | +1.85% | +1.65% | +2.12% | 13 |
| 16 | Attention | -1.82% | -1.45% | -1.65% | 13 |
| 16 | Residual | +18.42%| +21.15%| +19.85%| 13 |
| 17 | MLP | +4.52% | +4.15% | +4.85% | 13 |
| 17 | Attention | -3.55% | -3.12% | -3.25% | 13 |
| 17 | Residual | +31.55%| +35.80%| +33.12%| 13 |
| 18 | MLP | +8.03% | +7.72% | +8.55% | 39 |
| **18** | **Attention**| **-6.82%**| **-6.87%**| **-7.15%**| **39** |
| **18** | **Residual** | **+42.12%**| **+49.31%**| **+44.80%**| **39** |
| 19 | MLP | +9.45% | +9.12% | +9.85% | 13 |
| 19 | Attention | -2.15% | -1.85% | -2.05% | 13 |
| 19 | Residual | +46.80%| +54.20%| +48.95%| 13 |
| 20 | MLP | +10.54%| +14.23%| +11.25%| 39 |
| 20 | Attention | +1.32% | +0.30% | +1.15% | 39 |
| **20** | **Residual** | **+50.22%**| **+60.16%**| **+52.15%**| **39** |
| 21 | MLP | +11.85%| +13.50%| +12.45%| 13 |
| 21 | Attention | +0.45% | +0.38% | +0.55% | 13 |
| 21 | Residual | +51.85%| +60.80%| +53.40%| 13 |
| 22 | MLP | +12.40%| +14.20%| +13.10%| 13 |
| 22 | Attention | +0.25% | +0.18% | +0.32% | 13 |
| 22 | Residual | +52.40%| +61.20%| +54.10%| 13 |
| 23 | MLP | +13.80%| +15.60%| +14.50%| 13 |
| 23 | Attention | +0.12% | +0.08% | +0.21% | 13 |
| 23 | Residual | +52.85%| +61.45%| +54.60%| 13 |
| 24 | MLP | +15.04%| +17.87%| +15.80%| 39 |
| 24 | Attention | +0.00% | +0.60% | +0.25% | 39 |
| **24** | **Residual** | **+53.24%**| **+61.57%**| **+55.12%**| **39** |
| 25 | MLP | +13.50%| +15.10%| +14.20%| 13 |
| 25 | Attention | -0.45% | -0.32% | -0.38% | 13 |
| 25 | Residual | +51.10%| +59.80%| +52.80%| 13 |
| 26 | MLP | +10.20%| +11.80%| +11.05%| 13 |
| 26 | Attention | -0.85% | -0.65% | -0.72% | 13 |
| 26 | Residual | +48.50%| +56.40%| +50.15%| 13 |
| 27 | MLP | +6.40% | +7.15% | +6.85% | 13 |
| 27 | Attention | -1.25% | -0.95% | -1.10% | 13 |
| 27 | Residual | +42.15%| +49.80%| +44.20%| 13 |

---

## 付録C: 84サイト プローブ方向除去必要性検定・特異性 $Z_\perp$ 完全表 (Table A3)
*同一活性化部分空間内の50本ランダム直交方向との比較に基づく中和比率 $R_{\mathrm{neut}}$ および標準化特異性 $Z_\perp$*

| 層 | コンポーネント | 必要性スコア ($N_\ell$) | 中和比率 $R_{\mathrm{neut}}$ | 特異性 $Z_\perp$ | 帰無分布平均 $\mu_\perp$ | 帰無分布標準偏差 $\sigma_\perp$ | FDR補正後 $q_\perp$ | 特異的有意性 |
|---|---|---|---|---|---|---|---|---|
| 0 | MLP | 0.0078 | -0.139 | +0.604 | 0.0074 | 0.00067 | 0.857 | False |
| 0 | Attention | 0.0075 | -1.812 | +0.097 | 0.0074 | 0.00072 | 0.857 | False |
| 0 | Residual | 0.0082 | -2.825 | +1.146 | 0.0075 | 0.00063 | 0.857 | False |
| 5 | MLP | 0.0072 | -0.854 | -0.185 | 0.0073 | 0.00055 | 0.857 | False |
| 5 | Attention | 0.0076 | -1.125 | +0.320 | 0.0074 | 0.00061 | 0.857 | False |
| 5 | Residual | 0.0075 | -1.824 | +0.429 | 0.0072 | 0.00056 | 0.857 | False |
| 10 | MLP | 0.0081 | -0.412 | +0.884 | 0.0075 | 0.00062 | 0.857 | False |
| 10 | Attention | 0.0078 | -1.350 | -0.052 | 0.0078 | 0.00058 | 0.875 | False |
| 10 | Residual | 0.0079 | -0.920 | +0.315 | 0.0077 | 0.00059 | 0.857 | False |
| 14 | MLP | 0.0082 | -0.650 | +0.425 | 0.0079 | 0.00064 | 0.857 | False |
| 14 | Attention | 0.0080 | -1.150 | -0.115 | 0.0081 | 0.00055 | 0.875 | False |
| **14** | **Residual** | **0.0076** | **-1.542** | **+0.312** | **0.0074** | **0.00058** | **0.857** | **False** |
| **15** | **MLP** | **0.0084** | **-0.985** | **+0.124** | **0.0083** | **0.00062** | **0.875** | **False** |
| 15 | Attention | 0.0079 | -1.420 | -0.320 | 0.0081 | 0.00059 | 0.875 | False |
| 15 | Residual | 0.0081 | -0.450 | +0.550 | 0.0078 | 0.00054 | 0.857 | False |
| 18 | MLP | 0.0085 | -0.210 | +0.680 | 0.0081 | 0.00061 | 0.857 | False |
| **18** | **Attention**| **0.0079** | **-1.120** | **-0.215** | **0.0080** | **0.00055** | **0.875** | **False** |
| **18** | **Residual** | **0.0082** | **+0.145** | **+0.742** | **0.0078** | **0.00058** | **0.857** | **False** |
| **20** | **Residual** | **0.0080** | **+0.231** | **+0.518** | **0.0077** | **0.00055** | **0.857** | **False** |
| **24** | **Residual** | **0.0077** | **+0.185** | **+0.412** | **0.0075** | **0.00052** | **0.857** | **False** |
| 27 | Residual | 0.0083 | -0.320 | +0.189 | 0.0082 | 0.00059 | 0.875 | False |

*※注：全84サイトの最小 FDR $q_\perp$ は 0.8571 であり、プローブ方向除去による特異的有意差は全層で棄却された。*

---

## 付録D: 39ペア 代表18サイト詳細統計・IQR・95% Bootstrap CI (Table A5)

| サイト | コンポーネント | 有効ペア | Prompt平均 (中央値) | Gen平均 (中央値) | Gen IQR [25%, 75%] | 95% Bootstrap CI | 正の回復ペア率 |
|---|---|---|---|---|---|---|---|
| L10 | MLP | 39 | +0.42% (+0.43%) | +0.18% (+0.80%) | [-4.65%, +4.09%] | [-1.82%, +2.15%] | 53.8% |
| L10 | ATTN | 39 | +0.68% (+0.11%) | +0.91% (+0.94%) | [-2.26%, +3.49%] | [-0.85%, +2.65%] | 56.4% |
| L10 | RESID| 39 | +0.54% (+0.13%) | -1.42% (+0.15%) | [-3.84%, +3.29%] | [-3.45%, +0.62%] | 51.3% |
| L14 | MLP | 39 | +0.91% (+1.00%) | -1.55% (-1.05%) | [-4.31%, +2.22%] | [-3.25%, +0.15%] | 46.2% |
| L14 | ATTN | 39 | +0.15% (+0.28%) | -0.39% (+0.21%) | [-1.99%, +3.24%] | [-2.15%, +1.38%] | 51.3% |
| **L14**| **RESID**| **39** | **+1.38% (+1.07%)**| **+0.52% (+2.06%)**| **[-3.64%, +5.66%]**| **[-1.85%, +2.95%]**| **61.5%** |
| **L15**| **MLP** | **39** | **+0.51% (+0.47%)**| **-0.06% (+0.50%)**| **[-2.89%, +3.04%]**| **[-2.00%, +1.80%]**| **53.8%** |
| L15 | ATTN | 39 | -1.14% (-0.61%) | +3.46% (+5.32%) | [-1.70%, +12.66%]| [+0.45%, +6.48%] | 69.2% |
| L15 | RESID| 39 | +0.24% (+0.34%) | +7.40% (+9.70%) | [-1.43%, +16.36%]| [+3.15%, +11.65%]| 74.4% |
| L18 | MLP | 39 | +1.25% (+0.60%) | +8.03% (+7.72%) | [+2.52%, +15.68%]| [+4.85%, +11.20%]| 87.2% |
| **L18**| **ATTN**| **39** | **+0.44% (+0.68%)**| **-6.82% (-6.87%)**| **[-12.98%, +1.10%]**| **[-11.20%, -2.60%]**| **30.8%** |
| **L18**| **RESID**| **39** | **+0.63% (+0.28%)**| **+42.12% (+49.31%)**| **[+34.02%, +56.14%]**| **[+35.10%, +48.70%]**| **94.9%** |
| L20 | MLP | 39 | -0.42% (-0.44%) | +10.54% (+14.23%)| [+3.64%, +21.04%]| [+6.85%, +14.20%]| 82.1% |
| L20 | ATTN | 39 | +0.67% (+0.18%) | +1.32% (+0.30%) | [-6.25%, +6.94%] | [-1.45%, +4.10%] | 51.3% |
| **L20**| **RESID**| **39** | **+1.01% (+0.26%)**| **+50.22% (+60.16%)**| **[+32.47%, +66.31%]**| **[+42.60%, +57.50%]**| **97.4%** |
| L24 | MLP | 39 | -0.32% (-0.47%) | +15.04% (+17.87%)| [+5.99%, +25.96%]| [+10.85%, +19.20%]| 87.2% |
| L24 | ATTN | 39 | +0.34% (+0.09%) | +0.00% (+0.60%) | [-0.89%, +2.46%] | [-0.95%, +1.02%] | 56.4% |
| **L24**| **RESID**| **39** | **-0.26% (-0.19%)**| **+53.24% (+61.57%)**| **[+40.22%, +71.24%]**| **[+45.70%, +60.50%]**| **94.9%** |

---

## 付録E: 多層Residual同時介入の飽和特性 (Table A6)

| 組み合わせ条件 | 介入層数 | 平均OT回復率 (%) | 中央値OT回復率 (%) | L24単層に対する限界獲得量 |
|---|---|---|---|---|
| 単層: Layer 24 | 1 | 53.24% | 61.57% | 基準 (0.00%) |
| ペア: Layers 22, 24 | 2 | 53.98% | 62.10% | +0.74% |
| 3層: Layers 20, 22, 24 | 3 | 54.72% | 62.85% | +1.48% |
| 4層: Layers 20, 22, 24, 26 | 4 | 55.41% | 63.40% | +2.17% |
| 全後段: Layers 18–26 | 9 | 55.85% | 63.90% | +2.61% |

---

## 付録F: Ridge正則化スイープとMahalanobis OOD診断 (Table A7)

| $\log_{10}(\alpha)$ | 正則化パラメータ $\alpha$ | 活性化決定係数 $R^2$ | 線形 CKA | マハラノビス距離 $D_M$ | 多様体状態判定 |
|---|---|---|---|---|---|
| -1 | 0.1 | 0.891 | 0.948 | 48.92 | 深刻なOOD崩壊 |
| 0 | 1.0 | 0.887 | 0.945 | 44.91 | 中程度のOOD |
| 1 | 10.0 | 0.881 | 0.941 | 42.15 | 軽微なOOD |
| **2** | **100.0** | **0.874** | **0.936** | **40.82** | **最良トレードオフ点** |
| 3 | 1000.0 | 0.852 | 0.920 | 45.30 | 正則化収縮による乖離 |
| 4 | 10000.0 | 0.789 | 0.875 | 52.34 | 平均値への極端な崩壊 |
| *基準* | *自然なInstruct活性化* | *N/A* | *1.000* | **39.63** | *分布内リファレンス* |

---

## 付録G: LLaMA-3.2-1B-Instruct 追試プロファイル (Figure A10)

Llama-3.2-1B-Instruct（16層）の生成時局所因果回復率は、Layer 14 Residual において最大 1.82%、MLPで最大 0.95%（Layer 8）、Attentionで最大 1.25%（Layer 10）であり、全層にわたり平坦であった。これにより、Qwenで確認された後段Residualへの劇的な因果シフトは、トランスフォーマー固有の普遍法則ではなく、モデル規模やポストトレーニングの個別実装に依存する境界条件であることが確認された。

---

## 付録H: 再現スクリプト・生成データ・ハッシュ対応マニフェスト (Table A8)

| 論文内項目 | 実行スクリプトパス | 出力成果物ファイルパス | 検証対象・主要成果 |
|---|---|---|---|
| **Figure 1** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure1_overall_framework.png`, `.pdf` | 研究全体の概念図・中心仮説 |
| **Figure 2** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure2_four_panel_profile.png`, `.pdf` | 4パネル表現–因果層別プロファイル |
| **Figure 3** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure3_direct_peak_dissociation.png`, `.pdf` | 2大ピーク間の直接的対比・39ペア paired 可視化 |
| **Figure 4** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure4_greedy_collapse_vs_sensitivity.png`, `.pdf` | Greedy崩壊 vs 分布的感度 (Base vs Instruct) |
| **Fig. A1–A10**| `v3/scripts/plot_appendix_figures.py` | `v3/results/fig_a1_full_decodability.png` 等全10図 | 全層プロファイル・アブレーション・OOD診断・追試 |
| **Table 1** | `v3/scripts/expand_strict_dataset.py` | `v3/data/aipsy_strict_expanded.csv` | 感情語彙排除 Strict Expanded データセット |
| **Table 2, A5**| `v3/scripts/run_focused_39pairs_sweep.py`| `v3/results/focused_causal_sweep_39pairs.csv` | 代表18サイト 39ペア確証的介入評価 |
| **Table A1** | `v3/scripts/run_causal_localization_sweep.py`| `v3/results/causal_localization_sweep_joint_ot.csv` | 全28層 Prompt-Time スクリーニング原本 |
| **Table A2** | `v3/scripts/run_generation_time_causal_sweep.py`| `v3/results/generation_time_causal_sweep.csv` | 全28層 Generation-Time スクリーニング原本 |
| **Table A3** | `v3/scripts/run_probe_aligned_necessity_sweep.py`| `v3/results/probe_aligned_necessity_sweep.csv` | 全84サイト プローブ方向除去必要性検定原本 |
| **Table A6** | `v3/scripts/run_generation_multilayer_residual.py`| `v3/results/generation_multilayer_residual_results.csv` | 多層Residual同時介入の飽和特性データ |
| **Table A7** | `v3/scripts/run_ridge_alpha_sweep.py`| `v3/results/ridge_alpha_sweep_results.csv` | Ridge正則化掃引とマハラノビス距離原本 |
| **Table A8** | `v3/scripts/summarize_results.py` | `v3/results/summary_table.md` | 実験結果要約マニフェスト |


