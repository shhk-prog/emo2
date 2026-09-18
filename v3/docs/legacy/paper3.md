Decodability Does Not Localize Causal Leverage: 言語モデルにおける表現アクセス可能性と因果レバレッジの時空間的解離

要旨

大規模言語モデル（Large Language Models; LLMs）の内部活性化から、統語情報、知識、真偽、安全性、社会的属性、感情関連属性などを高精度に線形デコードできることが広く報告されている。しかし、ある属性が内部活性化から線形に読み出せることは、その活性化部位がモデル自身の下流計算において当該属性を強く制御する局所的因果点であることを意味しない。本研究では、この representation–use distinction を、Qwen2.5-1.5B-Instruct における affect-relevant representations をケーススタディとして体系的に検証する。

感情語彙などの表層交絡を制御した AIPsy-Affect Strict Expanded データセット（192 pair-id groups, 422 samples）を用い、81通りの離散 Valence–Arousal（VA）候補列に対する制約付き sequence-likelihood distribution、全28層・3コンポーネントの線形プロービング、matched activation substitution、probe-aligned projection ablation、および生成時介入を統合した。因果効果は、$9\times9$ VA 格子上の Manhattan ground cost に基づく2次元 Joint Optimal Transport（Joint OT）を用いて評価した。

第一に、affective peak 対 neutral 条件は中間層で最も強く線形デコード可能であり、MLPでは Layer 15（held-out $R^2=0.5610$）、Attentionでは Layer 18（$R^2=0.5495$）、Residual streamでは Layer 14（$R^2=0.5016$）で最大となった。

第二に、この高い線形アクセス可能性は、同じ部位における大きな局所因果レバレッジを意味しなかった。39組の held-out matched pairs を用いた評価では、最高解読部位である Layer 15 MLP の Prompt-time matched-substitution recovery は平均 $0.51%$（中央値 $0.47%$）、Generation-timeでも $-0.06%$（中央値 $0.50%$, 95% bootstrap CI $[-2.02%,1.83%]$）に留まった。また、Prompt-time全層探索では decodability と local recovery の間に強い単調関係は支持されなかった。

第三に、学習済み線形プローブ方向を活性化から射影除去しても、出力分布は系統的に neutral 条件へ近づかなかった。その変位は probe-orthogonal random directions の除去による変位と統計的に区別できず、全84サイトの Benjamini–Hochberg FDR 補正後に有意な部位は存在しなかった。

第四に、介入を自己報告生成直前へ移すと、局所因果レバレッジは後段 Residual stream に出現した。Layer 24 Residual の Generation-time recovery は $53.24%$（中央値 $61.57%$, 95% bootstrap CI $[45.74%,60.45%]$）に達した。最高解読部位 Layer 15 MLP と Layer 24 Residual の同一39ペア内の直接対比では、

[
\Delta G

G_{\mathrm{L24,Resid}}

G_{\mathrm{L15,MLP}}

53.30%
]

となり、95% bootstrap CI は $[45.34%,61.16%]$ であった。

したがって、本研究が示すのは、線形プロービングによる表現のアクセス可能性（accessibility）、matched substitution に対する局所因果レバレッジ（local causal leverage）、および probe direction の方向特異的必要性（direction-specific necessity）が、経験的に異なる性質であるということである。線形プロービングは情報を外部から読み取れる場所を同定するが、その情報がモデル自身の計算においてどこで、いつ、どの方向に沿って因果的に有効になるかを、それ自体では同定しない。

[
\boxed{\text{Decodability does not localize causal leverage.}}
]

⸻

1. はじめに

大規模言語モデルの内部表現を分析する標準的方法の一つとして linear probing が広く利用されている。ある層の hidden state から属性ラベルを線形分類器または線形回帰器によって高精度に予測できる場合、その属性に関する情報が当該活性化に線形アクセス可能な形で存在すると解釈される。

この方法は、統語構造、意味情報、知識、真偽、安全性、拒否挙動、感情極性、人格や社会的属性など、多様な対象に適用されてきた。しかし、ここには重要な解釈上の非対称性が存在する。Linear probe が直接示しているのは、

外部の読み出し器が、その活性化から情報を再構成できる

という事実である。

それは必ずしも、

モデル自身が、その場所で、その線形方向を用いて、その情報を下流行動の形成に利用している

ことを意味しない。

したがって、

[
\text{Linear Decodability}
\not\Rightarrow
\text{Local Causal Leverage}
]

という区別が必要である。

この representation–use distinction 自体は probing literature において以前から指摘されてきた。しかし、自己回帰型LLMにおいては、さらに「時間」および「多層分散ネットワーク」の問題が加わる。ある属性が Prompt 処理中に特定層から強く読み取れるとしても、その属性が実際に出力へ影響する計算は、応答生成時に別の層や別の表現状態へ再構成される可能性がある。また、初期の解釈可能性研究やプロトタイプ実験でしばしば採用される「4層おき（例: Layer 0, 4, 8, 12, 16, 20, 24, 27）」のような代表層の間引きサンプリングでは、情報が最も濃密に読み取れる層（中間層）と、出力生成を因果的に支配する層（浅層や深層）の局所性のズレを見落とす危険性がある。したがって、情報のアクセス可能性（Linear Probe）と因果的必要性（Ablation）の真の関係を解明するためには、特定の代表層の抽出にとどまらず、**全層・全サブモジュール（全28層 × 3コンポーネント = 計84サイト）を網羅的にプロービングおよびAblation / Substitution介入すること**が方法論的に不可欠である。

したがって、少なくとも以下を区別しなければならない。

1. Accessibility
    どこから情報を外部線形読み出し器が復元できるか（全層線形プロービング）。
2. Local causal leverage
    その部位の活性化を反実仮想的に置換したとき、下流出力がどの程度変化するか（全層置換介入）。
3. Direction-specific necessity
    プローブが同定した特定の線形方向を除去（Ablation）したとき、その情報に対応する出力が選択的に失われるか（全層射影Ablation）。
4. Temporal recruitment
    その因果的影響力が Prompt 処理時に存在するのか、あるいは自己回帰的な応答生成の途中で初めて現れるのか。

本研究では、この4つを同一のモデル、同一の刺激ペア、同一の出力分布上で直接比較する。

ケーススタディには affect-relevant textual conditions を用いる。ここで本研究は、LLMが主観的な感情経験を有することを主張するものではない。「first-person report」はあくまでモデルに要求した文法的・task-levelな出力形式を意味する。

**研究の動機付けと予備観察（Preliminary Observation）**:
本研究の発端として、まずフロンティア商用モデル（GPT-4o）において EmoBank（$N=3,210$）を用いた予備観察を実施した。その結果、他者読者の感情を推測する「認識（Recognition）タスク」では人間注釈値と極めて高い相関（Valence $r = 0.921$, Arousal $r = 0.590$）を示す一方で、テキストを読んだ後のモデル自身の感情を問う「自己報告（Post-reported VA）タスク」では 98.6% が完全な中立値 `{"valence": 5, "arousal": 5}` へ収束するという顕著な乖離が観測された。しかし、商用ブラックボックスモデルでは内部重みや活性化にアクセスできず、この現象が「出力層での表層的マスキング」なのか「内部表現自体の不反応」なのかを機械論的に検証することは不可能である。

**オープン重みモデル（4大ファミリー8モデル）による基盤的検証**:
内部解析へ移行するにあたり、最も重要な前提は「解析対象のモデルが、刺激の感情的文脈を正しく認識・符号化しているか」という認知的健全性（Sanity Check）である。そこで、オープン重みモデル（Qwen, Llama, Mistral, Gemma の主要4ファミリー計8モデル）を用い、EmoBank（$N=321$）および臨床統制データセット AIPsy-Affect（$N=144$）において認知的健全性と事後学習（Post-training）の影響を網羅的に検証した。その結果、以下の事実が実証された：
1. **感情認識の普遍性**: 複数ファミリーにおいて、事前学習（Base）および指示チューニング（Instruct）を通じて人間読者の感情判断を高精度に推定する能力（EmoBank $r_V \approx 0.6 \sim 0.88$）が獲得されている。
2. **事後学習効果のモデルファミリー依存性**: 事後学習の影響は一律の表層抑制（Suppression）ではない。Llama-Instruct では EmoBank 自己報告の **38.01%** が完全中立値 `(5, 5)` に収束する強い中立化傾向が見られる一方、Qwen, Mistral, Gemma では自己報告においても感情文脈と高く相関（$r_V = 0.56 \sim 0.86$）する表出を維持する。さらに AIPsy-Affect では、すべてのモデルにおいて Instruct 化に伴い感情条件と中立条件の弁別能（$|\Delta_V|$）が数倍〜9倍へと大幅に強化される。

したがって、事後学習は内部の感情表現を単に「消去・遮断」するのではなく、**内部表現から表層報告へのマッピング関係を再編成（Representation-to-report Remapping）** している。では、中間層で高精度にプロービング可能な内部感情表現は、下流の自己報告生成においてどのような因果的役割を果たしているのか？
$$\text{刺激文} \;\longrightarrow\; \underbrace{\text{Recognition (普遍的認識能: $r_V \approx 0.83$)}}_{\text{タスク認知的理解}} \;\longrightarrow\; \underbrace{\text{Internal Probe (高いDecodability: $R^2=0.561$)}}_{\text{中間層残差ストリームの符号化}} \;\longrightarrow\; \underbrace{\text{Remapping (表現-報告再編成)}}_{\text{ファミリー依存の表層マッピング}} \;\longrightarrow\; \underbrace{\text{Intervention (因果レバレッジの時空間分離)}}_{\text{Prompt時L15-19} \;\neq\; \text{生成時L24-27}}$$

本研究では、この認知的・幾何学的基盤を踏まえ、以下の研究質問（Research Questions）を設定する。

Research Questions

RQ1. 離散出力の挙動にかかわらず、内部の候補出力尤度分布（Sequence-Likelihood）において刺激依存的な感情構造はどのように保存されているか。

RQ2. Affect-related condition はモデル内部のどの層・コンポーネントから線形に読み出せるか（Decodability の同定）。

RQ3. 高い decodability を示す部位は、Prompt 処理時において大きな local causal leverage を示すか（Decodability と Causal Leverage の解離検証）。

RQ4. Probe が同定した線形方向は、下流出力に対して方向特異的に必要か（Direction-specific necessity の検証）。

RQ5. Local causal leverage は自己報告生成のどの時点・どの部位で出現するか（時空間的局在の特定）。

本研究の中心的な問いは RQ3–RQ5 である。

⸻

2. 関連研究 (Related Work)

本研究は、(i) linear probing による表現アクセス可能性の測定、(ii) activation patching による因果的介入、(iii) 線形表現仮説と concept erasure、(iv) LLM における感情・感情極性表現の研究という4つの先行研究群と関わる。以下ではそれぞれを概観し、本研究の新規性を位置づける。

2.1 Linear Probing とその方法論的限界

Linear probing は、モデルの中間表現からある属性を線形分類器・回帰器で予測できるかを測定する手法として、統語構造や意味情報の分析に広く用いられてきた。しかし、probing 手法には早くから方法論的な限界が指摘されている。Hewitt and Liang (2019) は、probe 自体が十分な表現力を持つ場合、probe がデータのランダムな統計的相関を記憶しているだけでも高い精度を示しうることを示し、control tasks という手法で probe の「選択性」（selectivity）を評価する枠組みを提案した。この研究は、probe 精度の高さそれ自体がモデル内部にその情報が「表現されている」ことの十分な証拠にはならないことを明確にした最初期の研究の一つである。

Belinkov (2021, 2022) によるサーベイは、probing classifiers の前提・欠点・改善の方向性を包括的に整理し、probing が示すのは「情報が表現から復元可能である」という事実に過ぎず、「モデルがその情報を実際にタスク遂行に使用している」ことは別の主張であると強調している。この区別は、しばしば representation–use distinction あるいは representation–utilization gap と呼ばれ、本研究が採用する Accessibility と Local Causal Leverage の区別の直接的な理論的先行研究である。

さらに Ravfogel et al. (2020, 2022) らによる concept erasure 研究では、学習された線形方向（あるいは非線形の nullspace）を除去した後もモデルの下流性能がある概念について変化しない、あるいは概念自体が別の非線形符号化を通じて残存する事例が報告されている。加えて、Kumar et al. (2022) による "Probing Classifiers Are Unreliable for Concept Removal and Detection" は、probe-derived direction の除去が概念の完全な除去を保証しないことを示し、probe 方向の因果的十分性に疑問を投げかけている。本研究の probe-aligned projection ablation と random-direction null control は、この系譜に属する検証手法である。

2.2 Activation Patching と Causal Tracing

Probing の限界を補うため、mechanistic interpretability 研究では、活性化を反実仮想的に置換する activation patching（causal tracing、interchange intervention とも呼ばれる）が標準的な因果的検証手法として確立している。この手法は、clean run・corrupted run・patched run の3段階比較により、特定のモデル構成要素が下流出力にどの程度因果的に影響するかを定量化する。

Meng et al. (2022) の ROME（Rank-One Model Editing）研究は、causal tracing によって事実関連付けの記憶が mid-layer MLP の主語トークン処理時に局在することを示し、その知見に基づいて rank-one weight editing で事実を書き換える手法を提案した点で、本研究に方法論的に最も近い先行研究の一つである。ROME の因果的発見は、「終端層の attention による情報のコピー」と「中間層 MLP による想起」という二段階の temporal/spatial な分離を示しており、本研究の Prompt-time と Generation-time の時間的分離という着想と構造的に類似する。ただし、ROME では主に単一事実に対する強い局在（高い recovery 率）が報告されている一方、本研究ではむしろ、高い decodability を示す部位が極めて低い local recovery を示すという逆方向の解離を報告する点で対照的である。

また、Hase et al. (2023) による "Does Localization Inform Editing?" ([PDF](https://asmadotgh.github.io/assets/pdf/13353_does_localization_inform_editi.pdf)) は、causal tracing によって特定された「重要な」層に対する編集が、必ずしも他の層への編集より効果的でないことを報告しており、localization の結果と editing 上の因果的有効性が一致しない場合があることを示している。この知見は、本研究の decodability（probing による localization の一種）と local causal leverage の解離という主張と一貫する。

Zhang and Nanda (2024) は、activation patching の実践における metric の選択（logit 差分の推奨）、corruption 戦略（Gaussian noise patching と symmetric token replacement の比較）、および 100% recovery を「完全な局在」、部分的 recovery を「分散計算」の証拠とみなす解釈規範を整理した、activation patching のベストプラクティスに関する体系的研究である。本研究の Joint Optimal Transport に基づく recovery 指標は、この枠組みにおける metric 選択の一つの拡張として位置づけられる。従来の activation patching 研究がスカラー logit 差分や単一トークンの確率を指標とするのに対し、本研究は81候補からなる離散 VA 分布全体の形状変化を、Manhattan ground cost を用いた2次元 Joint Optimal Transport で評価する。これは、属性が単一の yes/no 出力ではなく多次元的な分布として表現される場合の、より高解像度な因果効果測定を提供する。

2.3 因果的位置の時間的側面

本研究の Generation-time 解析に関連する先行研究として、causal tracing における「early site」と「late site」の区別が既に存在する。ROME の分析では、事実の想起は middle-layer MLP で生じる一方、その情報の出力への伝達は late layer の attention 機構によるコピー操作として分離されることが報告されている。しかし、これらの研究は主に Prompt 処理内での層間の時間的分離を扱っており、本研究のように、単一の Prompt-time 評価点と、自己回帰的な応答生成の途中（生成直前のトークン位置）という、生成プロセス全体を跨いだ時間的比較を体系的に行った研究は非常に限られている。

Inference-Time Intervention (ITI; Li et al., 2024) は、少数の attention head に沿った学習済み方向へ生成時に活性化をシフトすることで truthfulness を改善する手法であり、生成過程における介入の重要性を示した先行研究である。ITI は Prompt 処理後の生成過程全体にわたって介入を行う設計であるが、本研究のように、同一属性について Prompt-time と Generation-time 直前という異なる時点での matched-substitution recovery を直接対比し、両者の間の非単調な乖離を定量化した研究は、著者らの知る限り確認できない。

2.4 線形表現仮説とその反証

本研究の前提の一部は、線形表現仮説（linear representation hypothesis）、すなわち高次概念が活性化空間内の線形方向として符号化されるという仮説に依拠する。この仮説は、Representation Engineering (RepE; Zou et al., 2023) の枠組みにおいて、神経科学的な手法（LAT scan）を模した形で体系化され、Linear Artificial Tomography（LAT）による刺激設計・神経活動収集・線形モデル構築という三段階の手続きが提案された。RepE はさらに、線形方向をトップダウンに操作することでモデル行動を制御する representation control 手法を提示している。

一方、この仮説に対する反証的な研究群も蓄積されている。"On the Failure of a Universal Linear Representation Hypothesis in Deep Neural Networks" は、ある関数族に対して線形方向による概念復元が理論的に不可能であることを情報理論的下界とともに示し、"On the Limits of Linear Representation Hypotheses in Large Language Models" は、カオス的動力学の下で軌道が指数的に分岐する条件下では線形表現が破綻することを論じている。本研究の probe-aligned direction ablation が random orthogonal direction と統計的に区別できなかったという結果は、これらの反証研究と整合的であり、少なくとも Qwen2.5-1.5B-Instruct の affect-relevant representation については、probe が同定した特定の線形方向が、モデル自身の計算にとって特異的に必要な因果方向ではなかったことを支持する。

2.5 感情・感情極性のLLM内部表現に関する研究

LLM の感情関連表現を対象とした先行研究も近年増加している。Di Palma et al. (2025) ([ACL 2025](https://aclanthology.org/2025.acl-long.306/)) は、probe classifier を用いて複数の LLM における感情極性の layer-wise encoding を分析し、感情情報が mid-layer に最も集中していることを報告した。この結果は、本研究が Qwen2.5-1.5B-Instruct の MLP Layer 15、Attention Layer 18、Residual Layer 14 において最大 decodability を観測した知見と方向性として一致する。

また、Maheswaran and Desarkar (2026; EACL 2026) による "A Unified View on Emotion Representation in Large Language Models" は、複数の LLM 間で共通する感情表現が later layer に存在し、感情の方向を捉えるベクトルが感情理解タスクと関連することを報告している。さらに、"Emotion Beyond Language: Probing Multilingual..." と題する研究は、20の感情カテゴリと13言語にわたる instruction-tuned LLM の感情表現を probing により分析している。

近年の研究の多くは主に probing や表現類似性を通じて感情表現の特徴づけを行っているが、層・コンポーネント・生成時点を網羅した matched-substitution による体系的な因果的局所化の検証は比較的未開拓の領域である（Several recent studies primarily characterize emotion representations through probing or representation similarity; systematic matched-substitution localization across layer, component, and generation time remains comparatively underexplored）。本研究は、これらの感情表現研究と同一の Qwen ファミリーモデルを対象としつつ、probing 結果を出発点ではなく前提条件として位置づけ、その先の因果的検証を主目的とする点で異なる。

2.6 本研究の新規性と位置づけ

以上の先行研究を踏まえると、本研究の新規性は以下の5点に整理できる。

1. **Accessibility、Local Causal Leverage、Directional Necessity の体系的対比:** 多くの probing 研究（Belinkov, 2021; Di Palma et al., 2025 など）は representation–use distinction を理論的に指摘するか、単一属性での decodability を報告するに留まり、同一モデル・同一属性・同一評価指標上で accessibility、local causal leverage、direction-specific necessity の三者を定量的に対比した研究は限られている。本研究は、これら三つの性質を、全28層・3コンポーネントにわたる系統的な sweep として直接比較し、decodability 最大部位と causal recovery 最大部位が一致しないことを、単一の効果量対比として定量的に提示する。
2. **生成過程を跨ぐ「時間的動員（Temporal Recruitment）」の実証:** ROME をはじめとする causal tracing 研究は主に Prompt 処理内での時間的・空間的局在を報告してきた一方、本研究は Prompt-time と Generation-time 直前という、自己回帰的生成プロセスを跨いだ二時点の比較を行い、同一部位・同一属性における因果的レバレッジの「時間的動員」（temporal recruitment）を報告する。特に Layer 15 MLP が両時点でほぼゼロの回復率を示す一方、Layer 24 Residual が生成時のみ大きな回復率（$53.24\%$）を示すという結果は、既存の early/late site 分離の知見を、単一時点の Prompt 処理内比較から生成過程全体をカバーする比較へと拡張するものである。
3. **離散二次元分布に対する 2D Joint Optimal Transport の導入:** 本研究は recovery 指標として、単一トークンの確率や logit 差分ではなく、81候補からなる離散 Valence–Arousal 分布全体に対する Manhattan ground cost を用いた 2次元 Joint Optimal Transport を採用している。これは、activation patching のベストプラクティス研究が主に単変量の出力指標を前提としているのに対し、多次元的な属性空間における因果効果をより高解像度に評価する試みである。
4. **Random Direction Null Control と FDR 補正による方向特異性の厳格検証:** probe-aligned direction ablation において、probe 方向の除去効果を probe-orthogonal random direction の empirical null distribution と比較し、Benjamini–Hochberg FDR 補正を適用した統計的検証は、線形表現仮説への反証研究や concept erasure 研究の知見を、感情関連属性という新たな対象に対して多重比較を制御した形で適用する試みである。
5. **Cross-Family Partial Replication によるモデル依存性の同定:** cross-family partial replication（Qwen2.5-1.5B-Instruct から Llama-3.2-1B-Instruct への追試）により、観測された時間的・空間的局在パターンがモデル依存的であり、単一モデルでの結果を過度に一般化すべきでないことを明示的に示した点も、感情表現研究の多くが単一モデルまたは同一ファミリー内での分析に留まる中で、本研究が付加する頑健性検証の一側面である。

総じて、本研究は既存の probing 限界の理論的指摘、activation patching の方法論、線形表現仮説への反証、感情表現の probing 研究という複数の系譜を統合し、それらが個別に論じてきた「decodability と因果的使用は異なる」という命題を、単一のモデル・単一の属性・複数の時点・複数の介入様式にわたって定量的に実証した点に新規性がある。

⸻

3. 操作的定義

3.1 Linear Accessibility

層 $\ell$、コンポーネント $c$ の活性化 $h_{\ell,c}$ から affective peak 対 neutral 条件を held-out data 上で線形予測する性能を、

[
D_{\ell,c}
]

とする。

本研究では Ridge regression の held-out $R^2$ を用いる。

$D_{\ell,c}$ が高いことは、その活性化に当該条件に関する線形アクセス可能な情報が存在することを意味する。

これは当該部位が因果的ボトルネックであることを意味しない。

3.2 Local Causal Leverage

本研究における local causal leverage は、ある部位が情報の起源であることや、唯一の必要経路であることを意味しない。

層 $\ell$、コンポーネント $c$、トークン位置 $t$ に対し、

[
do
\left(
h_{\ell,c,t}^{\mathrm{neutral}}
\leftarrow
h_{\ell,c,t}^{\mathrm{peak}}
\right)
]

という matched substitution を行う。

このとき、パッチ後の出力分布が neutral baseline から matched peak target へどの程度移動したかを local causal recovery と定義する。

したがって、本研究でいう local causal leverage とは、

tested matched-substitution intervention に対する下流出力分布の感受性

である。

局所回復が小さいことは、当該部位が大域的に非因果的であることを意味しない。分散表現、冗長経路、複数トークンにまたがる計算、非線形相互作用などの可能性は残る。

3.3 Probe-Aligned Direction-Specific Necessity

学習済み probe weight の単位方向を $\hat v_{\mathrm{probe}}$ とし、

[
h’

h

(h^\top \hat v_{\mathrm{probe}})
\hat v_{\mathrm{probe}}
]

によってその1次元成分を除去する。

この操作が出力を neutral target 方向へ選択的に移動させ、かつ probe-orthogonal random directions の除去より大きな効果を示す場合、その方向について局所的な direction-specific necessity が支持される。

⸻

4. モデルとデータ

4.1 Models

主解析には、

* Qwen/Qwen2.5-1.5B
* Qwen/Qwen2.5-1.5B-Instruct

を用いる。

Qwen2.5-1.5B は28 Transformer layersを持ち、本研究では Layer 0–27 を解析する。

中心となる within-model causal analysis は Instruct モデルを対象とする。

Base/Instruct comparison は post-training の一般的因果効果を推定するためではなく、representational alignment と functional equivalence を区別する補助的 stress test として用いる。

また、cross-family exploratory replication として Meta Llama-3.2-1B-Instruct を用いる。

4.2 AIPsy-Affect Strict Expanded

主要な機構実験には AIPsy-Affect Strict Expanded を用いる。

データセットは、

* 192 pair-id groups
* 422 samples

から構成される。

各 group は narrative structure、characters、tense、lexical complexity などを可能な限り保持したまま、affective intensity を変化させた matched stimuli からなる。

主条件は、

* Neutral
* Moderate
* Peak

である。

明示的な感情語彙への依存を抑えることで、単純な lexical cue detection が probe performance を支配する可能性を低減する。

Data split

pair-id group 単位で以下のように分割する。

Split	Groups	Samples	Usage
Train	76	169	Linear probe fitting
Alignment-dev	58	124	Base→Instruct mapping / statistics
Held-out test	58	129	Final probing / causal intervention

Held-out test には39組の complete Peak–Neutral matched pairs が存在する。

全84サイトを対象とする探索的全層スクリーニングでは、計算量制約から事前に固定した15組を使用する。

代表層の効果量推定には39組すべてを使用する。

### Table 1: 実験設計およびデータセット諸元 (Dataset and Experimental Design Specifications)

| 項目 | 仕様 | 方法論的意義・役割 |
|---|---|---|
| **主対象モデル** | Qwen2.5-1.5B / Instruct | 28層トランスフォーマー, 隠れ層次元 $d=1536$ |
| **比較対照モデル** | Llama-3.2-1B-Instruct | 16層トランスフォーマー; アーキテクチャ間追試 |
| **使用データセット** | AIPsy-Affect Strict Expanded | 感情語彙を完全排除し、文長・構文・ドメインを厳密整合 |
| **総サンプル規模** | 192ペアグループ / 422サンプル | 訓練: 76群 (169), 開発: 58群 (124), テスト: 58群 (129) |
| **評価コホートペア数** | 39完全テストペア | 感情文と中立文が1対1で対応する厳密バランスコホート |
| **介入対象サイト** | 28層 $\times$ 3コンポーネント = 84サイト | MLP出力、Attention出力、Residual stream |
| **自己報告タスク** | 一人称VA形式のJSON出力 | `{"valence": V, "arousal": A}` |
| **評価スコア空間** | 81通りの離散VA候補列 | 全候補列に対する拘束シーケンス対数尤度正規化 |
| **主因果距離指標** | 2D Joint Optimal Transport cost | $9 \times 9$ VA格子上のManhattan ground costに基づく最適輸送コスト |
| **全層探索スクリーニング** | 15テストペア（全84サイト網羅） | 探索的全層プロファイル探索 (Exploratory full-layer screen) |
| **代表層集中評価** | 39テストペア（代表18サイト） | 全数コホート効果量評価 (Focused full-cohort effect-size evaluation) |

⸻

5. 制約付きSequence-Likelihood Evaluation

5.1 Candidate space

Valence $v$ および Arousal $a$ を、

[
v,a\in{1,\ldots,9}
]

とし、全81候補

[
y_{v,a}

\texttt{{“valence”: v, “arousal”: a}}
]

を構成する。

5.2 Conditional sequence likelihood

入力 $x$ に対する候補のスコアは、

[
s_{v,a}

\sum_{t=1}^{|y_{v,a}|}
\log
P(y_{v,a,t}\mid x,y_{v,a,<t})
]

とする。

候補集合上で、

[
P(v,a\mid x)

\frac{\exp(s_{v,a})}
{\sum_{v’,a’}\exp(s_{v’,a’})}
]

を計算する。

長さによる影響を確認するため、

[
s_{v,a}^{\mathrm{norm}}

\frac{s_{v,a}}{|y_{v,a}|}
]

を用いる robustness analysis も実施する。

期待値は、

[
E[V\mid x]

\sum_{v,a}vP(v,a\mid x)
]

[
E[A\mid x]

\sum_{v,a}aP(v,a\mid x)
]

として計算する。

⸻

6. Joint Optimal Transport による因果回復率

81候補を $9\times9$ の VA grid とみなす。

二つの分布 $P,Q$ に対し、

[
\mathrm{OT}_{VA}(P,Q)

\min_{\gamma\in\Pi(P,Q)}
\sum_{i,j}
\gamma_{ij}c(i,j)
]

を計算する。

Ground cost は、

[
c((v,a),(v’,a’))

|v-v’|
+
|a-a’|
]

という Manhattan distance とする。

したがって、本稿ではこれを 2D Joint Optimal Transport cost または Joint OT distance with Manhattan ground cost と呼ぶ。

Matched Peak–Neutral pair に対する回復率を、

[
R

1-
\frac{
\mathrm{OT}_{VA}
(P_{\mathrm{patch}},P_{\mathrm{peak}})
}{
\mathrm{OT}_{VA}
(P_{\mathrm{neutral}},P_{\mathrm{peak}})
}
]

と定義する。

$R>0$ は patched distribution が Peak target へ近づいたことを意味し、$R=0$ は改善なし、$R<0$ は Peak target からさらに離れたことを意味する。

負の回復率は、それ自体から inhibitory mechanism を意味しない。Off-manifold perturbation や context incompatibility も原因となり得る。

不安定な比率を防ぐため、

[
\mathrm{OT}_{VA}
(P_{\mathrm{neutral}},P_{\mathrm{peak}})
<
0.05
]

のペアは recovery analysis から除外する。

⸻

7. Statistical Analysis

全層スクリーニングは localization を目的とする exploratory analysis とし、代表層39-pair evaluation を効果量の安定性評価として扱う。

平均 recovery の95%信頼区間は matched pair を resampling unit とする non-parametric bootstrap により算出する。

主たる peak-site contrast は各ペア内で、

[
\Delta G_i

G_{i,\mathrm{L24,Resid}}

G_{i,\mathrm{L15,MLP}}
]

を計算し、その平均差およびbootstrap confidence intervalを報告する。

Layerwise association には Spearman rank correlation を用いる。

Probe-aligned necessity に関する多重比較には Benjamini–Hochberg FDR correction を適用する。

統計的非有意性は効果の不存在の証明とは解釈せず、effect size と confidence interval を併記する。

⸻

8. Results

8.1 Greedy Collapse Does Not Imply Distributional Invariance

Qwen2.5-1.5B-Instruct の greedy first-person report は98.6%の試行で、

[
(V,A)=(5,5)
]

に集中した。

しかし、81候補の likelihood distribution から得られる expected Valence は刺激間で変化した。

EmoBank reader Valence との相関は、

[
r=0.629,
\qquad
\rho=0.618
]

であった。

一方、Instructモデルでは expected Valence の dynamic range は Base より強く圧縮された。

したがって、

[
\boxed{
\mathrm{Greedy\ Collapse}
\neq
\mathrm{Distributional\ Invariance}
}
]

である。

ここから分かるのは、モデルに subjective emotion が存在することではない。

固定 candidate set 上の conditional output distribution が刺激依存的であるということである。

Figure 1

Greedy Collapse vs. Distributional Sensitivity

Panel A: Base/Instruct における $(5,5)$ greedy output rate。

Panel B: Human reader Valence と expected $E[V]$ のscatter plot。

⸻

8.2 Intermediate Layers Maximally Encode the Affective Condition

Prompt final token において全28層・3コンポーネントから activation を抽出し、Peak 対 Neutral 条件を予測する Ridge probe を学習した。

各componentの最大 held-out $R^2$ は、

[
R^2_{\mathrm{MLP},15}

0.5610
]

[
R^2_{\mathrm{ATTN},18}

0.5495
]

[
R^2_{\mathrm{RESID},14}

0.5016
]

であった。

したがって affective condition は、特に中間層において外部線形読み出し器から高い精度でアクセスできる。

ここで decodability maximum を、

[
(\ell_D^,c_D^)

\arg\max_{\ell,c}
D_{\ell,c}
]

と定義すると、

[
(\ell_D^,c_D^)

(15,\mathrm{MLP})
]

である。

⸻

8.3 High Decodability Does Not Imply Prompt-Time Local Recovery

次に、Prompt final token において Peak donor activation を matched Neutral run へ置換した。

15-pair exploratory full-layer screen では、最大平均 Joint OT recovery は、

* MLP: Layer 10, $2.20%$
* Attention: Layer 20, $1.48%$
* Residual: Layer 16, $1.68%$

であり、全体として極めて小さかった。

特に decodability maximum である Layer 15 MLP では、探索screenにおける recovery は約 $1%$ に留まった。

全層 decodability と Prompt-time recovery の Spearman correlation は、

[
\rho_{\mathrm{MLP}}

0.296,
\qquad
p=0.127
]

[
\rho_{\mathrm{ATTN}}

0.023,
\qquad
p=0.908
]

[
\rho_{\mathrm{RESID}}

-0.039,
\qquad
p=0.842
]

であった。

28層という標本数を考慮すれば、この結果を「無相関の証明」と解釈すべきではない。しかし、少なくとも強い単調対応関係を支持する証拠は得られなかった。

代表6層を39 complete matched pairs で再評価すると、Layer 15 MLP の Prompt-time recovery は、

[
0.51%
]

中央値、

[
0.47%
]

であった。

つまり、

[
D_{\mathrm{L15,MLP}}

0.561
]

という全サイト最大の decodability が存在しても、

[
S_{\mathrm{L15,MLP}}
\approx0
]

であった。

⸻

8.4 Probe-Aligned Direction Removal Shows No Specific Local Necessity

次に、Probe が学習した1次元方向自体の因果的重要性を検証した。

各サイトで、

[
h’

h-(h^\top \hat v_{\mathrm{probe}})
\hat v_{\mathrm{probe}}
]

として probe direction を除去した。

15-pair全層スクリーニング（全84サイト、各サイト20本の直交ランダム方向との比較、最小 raw $p = 1/21 = 0.0476$）では、出力分布の変位は小さく、Neutralization ratio も一貫して0近傍であった。

さらに、probe direction に直交する random directions を除去した場合の empirical null distribution と比較した。

84サイトに Benjamini–Hochberg FDR correction を適用すると、最小 $q = 0.857$ であり有意な特異的必要性を示すサイトは存在しなかった（全サイト $q \ge 0.857$）。

高解像度controlとして代表5層・15サイトに対して100本の orthogonal random directions を用いた場合にも、

[
p_{\perp}
\ge0.297
]

であり、全サイトで、

[
q=1.000
]

となった。

したがって、

[
\boxed{
\text{No evidence for probe-aligned local necessity}
}
]

である。

これは probe direction がモデル全体で非因果的であるという主張ではない。

より限定的に、

Tested local activation slice において、その1次元線形方向を除去しても、出力の方向特異的な中和は生じなかった。

という結果である。

⸻

8.5 Strong Local Causal Leverage Emerges During Generation

Prompt final token では局所回復がほぼ観察されなかったため、介入時点を変更した。

モデルに自己報告prefix、

{"valence":

を与え、その最終トークン位置で matched Peak activation を Neutral run へ移植した。

39 complete test pairs を用いた representative-site evaluation では、結果が大きく変化した。

Layer 15 MLP

[
D=0.561
]

に対して、

[
G_{\mathrm{L15,MLP}}

-0.06%
]

Median:

[
0.50%
]

95% bootstrap CI:

[
[-2.02%,1.83%]
]

であった。

したがって、最高decodability siteは Generation-time においてもほぼゼロの局所回復しか示さなかった。

一方、late Residual stream では、

[
G_{\mathrm{L18,Resid}}

42.12%
]

[
G_{\mathrm{L20,Resid}}

50.22%
]

[
G_{\mathrm{L24,Resid}}

53.24%
]

へ急増した。

Layer 24 Residual の中央値は、

[
61.57%
]

95% bootstrap CI は、

[
[45.74%,60.45%]
]

であった。

Prompt-time の Layer 24 Residual decodability は、

[
D_{\mathrm{L24,Resid}}

0.147
]

と比較的低かった。

すなわち、

[
D_{\mathrm{L15,MLP}}

0.561,
\qquad
G_{\mathrm{L15,MLP}}
\approx0
]

である一方、

[
D_{\mathrm{L24,Resid}}

0.147,
\qquad
G_{\mathrm{L24,Resid}}

0.5324
]

である。

この結果は、decodability maximum と local causal-leverage maximum が異なることを直接示している。

⸻

8.6 Direct Peak-Site Contrast

39ペアについて Layer 15 MLP と Layer 24 Residual の generation-time recovery を直接比較した。

[
\Delta G_i

G_{i,\mathrm{L24,Resid}}

G_{i,\mathrm{L15,MLP}}
]

とすると、

[
\overline{\Delta G}

53.30%
]

となった。

95% bootstrap confidence interval は、

[
[45.34%,61.16%]
]

であり、ゼロを大きく上回った。

この contrast は、本研究の中心命題を支える主要な effect-size evidence である。

つまり、

[
\boxed{
\arg\max D_{\ell,c}
\neq
\arg\max G_{\ell,c,t}
}
]

である。

より具体的には、

[
\boxed{
\text{Decodability maximum}

\mathrm{L15\ MLP}
}
]

に対して、

[
\boxed{
\text{Strongest evaluated generation-time recovery}

\mathrm{L24\ Residual}
}
]

であった。

### Table 2: 39完全テストペアにおける代表サイト結果サマリー (Main Representative-Site Results)

| 介入サイト | コンポーネント | プローブ $R^2$ ($D_\ell$) | Prompt回復率 ($S_\ell$) | Generation回復率 ($G_{\ell,t}$) | 95% Bootstrap CI | 因果的位置づけ |
|---|---|---|---|---|---|---|
| **Layer 14** | Residual | 0.502 | 1.38% (中央値 1.07%) | 0.52% (中央値 2.06%) | [-3.64%, +5.66%] | 残差初期ピーク; 局所因果回復は微小 |
| **Layer 15** | MLP | **0.561** | **0.51%** (中央値 0.47%) | **-0.06%** (中央値 0.50%) | **[-2.00%, +1.80%]** | **全層最高解読部位; tested local intervention下で局所回復示さず** |
| **Layer 18** | Attention | 0.550 | 0.44% (中央値 0.68%) | -6.82% (中央値 -6.87%) | [-11.20%, -2.60%] | 注意ピーク; 生成時置換は負の回復 |
| **Layer 18** | Residual | 0.485 | 0.63% (中央値 0.28%) | 42.12% (中央値 49.31%) | [+35.10%, +48.70%] | 後段残差レバレッジの発現点 |
| **Layer 20** | Residual | 0.401 | 1.01% (中央値 0.26%) | 50.22% (中央値 60.16%) | [+42.60%, +57.50%] | 強い因果伝達チャネル |
| **Layer 24** | Residual | **0.147** | -0.26% (中央値 -0.19%) | **53.24%** (中央値 61.57%) | **[+45.70%, +60.50%]** | **最大因果レバレッジ部位; 線形解読能は低水準** |

Figure 2

Four-Panel Representational–Causal Profile

* Panel A: Full-layer decodability
* Panel B: Prompt-time matched-substitution recovery
* Panel C: Probe-aligned necessity / orthogonal-null comparison
* Panel D: Generation-time matched-substitution recovery

Figure 3

Direct Decodability–Causal Leverage Dissociation

左側に L15 MLP と L24 Residual の $D$ と $G$ を対比。

右側に39 matched pairsの paired recovery distribution を表示する。

⸻

9. Secondary and Boundary Analyses

9.1 Multi-Layer Late Residual Patching

Late residual recovery が複数層の独立した加算効果によるものかを検証するため、Generation-time に複数のResidual sitesを同時に置換した。

独立した multilayer pipeline では、

* L18: $43.51%$
* L20: $51.85%$
* L24: $55.08%$
* L20+L24: $55.67%$
* L18+L20+L24: $55.74%$
* L18–24: $55.35%$

であった。

Layer 24 単独から7層同時置換へ拡大しても回復率はほとんど増加しなかった。

この結果は、late Residual sites が完全に独立した加算的因果寄与を持つという仮説とは整合しない。

むしろ、

* redundant transmission,
* downstream saturation,
* intervention-state dependence

などと整合する。

ただし、本実験だけではこれらを区別できない。

⸻

9.2 Cross-Model Predictive Alignment Does Not Ensure Natural Activations

Base→Instruct representational mapping の補助実験では、Ridge alignment によって一定の predictive alignment が得られた。

一方、mapped representations の Mahalanobis radius は natural Instruct activation と大きく異なった。

Natural Instruct activation の中央値は、

[
D_M=39.63
]

である一方、mapped activations は弱正則化条件でも、

[
D_M\approx9.8
]

に収縮した。

したがって、

[
\boxed{
\text{Predictive Alignment}
\neq
\text{Distributional Typicality}
}
]

である。

Cross-model patching の失敗だけから functional decoupling を断定できない理由の一つである。

この分析は本論文の中心証拠ではなく、cross-model causal interpretation に対する補助的 caution と位置づける。

⸻

9.3 Cross-Family Partial Replication

Meta Llama-3.2-1B-Instruct に対し、Generation-time matched-substitution screen を実施した。

Qwen2.5-1.5B-Instruct で観察された強い positive late-residual recovery は、Llama の初期追試では再現されなかった。

Llama では、

* MLP output: 回復率は全層で一貫して負値またはほぼ0%であり、最大値は Layer 15 の -0.36%、
* Attention output: 正の最大回復率は Layer 4 Attention の +0.73%（The largest positive recovery was 0.73% at Layer 4 Attention）、
* Residual stream: Layer 0（-36.18%）から Layer 15（-4.23%）に至る全16層で一貫して負値（中央値 -18.7%）の recovery

となった。

したがって、

Qwenで観察されたlate-residual localizationは、今回の初期Llama partial replicationでは再現されなかった。

と言える。

これは、

[
\text{late-residual localization is universal}
]

という主張を支持しない。

一方、Llamaでは本研究と同一条件で full decodability analysis を実施していないため、より一般的な

[
\text{decodability}
\neq
\text{causal leverage}
]

という命題そのもののcross-family replicationとはみなさない。

⸻

10. Discussion

10.1 Accessibility and Causal Leverage Are Different Empirical Properties

本研究で最も重要なのは、Linear probe の有用性を否定することではない。

Linear probe は、

ある表現から外部観測者が情報を読み出せるか

を測定する。

Matched activation substitution は、

その表現スライスを反実仮想条件へ交換したとき、下流出力が変わるか

を測定する。

Projection ablation は、

Probeが学習した特定線形方向が、その場所で選択的に必要か

を測定する。

これらは認識論的にも方法論的にも全く異なる問いである。Linear Probe が「情報が内部状態から線形に読み出し可能か（Accessibility/Decodability）」という相関的・受動的な観察を行うのに対し、Ablation は「その情報をモデルから消去したときに下流計算が破綻するか（Necessity）」という能動的介入を行う。したがって、「Ablation もプロービングと同様に全層・全モジュールで行うべきか」という問いに対しては、**全層網羅的な Ablation スイープこそが方法論的に必須である**と結論される。もし一部の中間層のみで Ablation を行った場合、残差接続によるバイパスや他層での分散的補償を見誤り、「プローブで読めるからこの層が必要だ」あるいは「この層を消しても効かないから情報は無意味だ」という誤った推論に陥るためである。

Qwen2.5-1.5B-Instructにおいて全84サイトを網羅した結果、

[
\boxed{
\text{Accessibility},
\quad
\text{Local Causal Recovery},
\quad
\text{Direction-Specific Necessity}
}
]

が一致しなかった。

特に、

[
D_{\mathrm{L15,MLP}}

0.561
]

であるにもかかわらず、

[
S_{\mathrm{L15,MLP}}

0.0051
]

かつ、

[
G_{\mathrm{L15,MLP}}
\approx0
]

であった。

したがって、高い decodability は tested intervention family の下で strong local control knob を意味しなかった。

⸻

10.2 Causal Leverage Is Temporally Recruited

本研究が単純な「probeとcausalityは違う」という再確認に留まらない点は、Generation-time analysis にある。

Prompt-timeではほぼ存在しなかった強い matched-substitution recovery が、自己報告生成直前には late Residual stream に出現した。

これは、

[
\boxed{
\text{Causal leverage is position- and time-dependent}
}
]

という可能性を示す。

つまり、情報は中間層ですでに外部から読み取れるものの、その情報が出力形成に強く作用する形へ再構成されるのは、より後段かつ生成時である可能性がある。

ただし、この結果から、

L24 Residual が感情情報の起源である

とは言えない。

Residual stream は上流計算の累積状態であり、そこで大きな intervention sensitivity が観測されることは、その情報がそこで初めて生成されたことを意味しない。

本研究が示すのは、あくまで、

tested local matched-substitution intervention に対する最大の下流感受性が、その時点・その部位に現れた

ということである。

⸻

10.3 Decodability Does Not Identify the Causal Direction

Probe direction removal の結果も重要である。

もし linear probe の weight vector がモデル自身の因果的「感情軸」であるなら、その方向を除去すれば output distribution が Neutral 側へ系統的に移動すると期待される。

しかし、そのような傾向は得られなかった。

さらに、その変位は probe-orthogonal random direction removal と区別できなかった。

したがって、

[
\boxed{
\text{A predictive linear direction need not be a causal control direction}
}
]

である。

この結果は linear representation engineering を否定するものではない。

Addition steering と deletion-based necessity は異なる intervention であり、ある方向を強く外挿することで行動を動かせても、その方向が natural computation において必要とは限らない。

⸻

10.4 Implications for Mechanistic Interpretability

本研究の結果は、LLM内部表現を解析する際、少なくとも以下の4軸を分離する必要性を示す。

[
\boxed{
\begin{aligned}
&\textbf{Accessibility:}
&&\text{Can an external decoder read the information?}\
&\textbf{Local leverage:}
&&\text{Does matched substitution change the output?}\
&\textbf{Directional necessity:}
&&\text{Is the probe-aligned direction specifically required?}\
&\textbf{Temporal recruitment:}
&&\text{When does causal sensitivity emerge?}
\end{aligned}
}
]

Linear probing は第一の問いに強い。

しかし第二から第四を測定するには、因果介入が必要である。

したがって、

「Layer $\ell$ に concept X が表現されている」

という表現を、

「Layer $\ell$ から concept X を外部線形読み出し器で高精度に復元できる」

と限定して記述することが望ましい。

さらに因果的主張を行う場合には、substitution、ablation、temporal intervention、random-direction control などを併用する必要がある。

### Table 3: 研究の問い・方法論的証拠・実証結果・理論的示唆の総括対照表 (Claims and Evidence)

| 研究上の問い | 方法論的証拠 | 得られた実証結果 | 理論的示唆・結論 |
|---|---|---|---|
| **RQ1: 情動情報は内部に保持されているか？** | 81候補列拘束シーケンス尤度評価 | Greedy出力崩壊（98.6%）と高い尤度相関（$r=0.629$）が共存。 | **Greedy Collapse $\neq$ Distributional Invariance**。出力表層の不変性は内部分布の感度消失を意味しない。 |
| **RQ2: 情動情報はどこで線形解読可能か？** | 全28層 $\times$ 3コンポーネント線形プロービング ($D_\ell$) | 中間層（L15 MLP: $R^2=0.561$, L18 Attn: $R^2=0.550$, L14 Resid: $R^2=0.502$）で極大化。 | 情動情報は中間表現において外部線形読み出しに対して強くアクセス可能。 |
| **RQ3: 解読能が高い部位は因果レバーを持つか？** | プロンプト最終トークンにおけるペアマッチ置換 ($S_\ell$) | L15 MLP回復率は 0.51%（中央値 0.47%）。全層相関 $\rho \le 0.296$ ($p > 0.05$)。 | **Decodability does not localize prompt-time causal leverage**。高解読部位は因果的ボトルネックではない。 |
| **RQ4: プローブ方向ベクトルは因果的に必要か？** | 幾何学的直交射影除去 ($N_\ell$, $Z_\perp$) | 84サイト全数においてランダム直交方向と有意差なし（No evidence for probe-aligned local necessity; 全て FDR $q \ge 0.857$, $N=100$ で $q=1.000$）。 | プロービングが同定する読み出し軸は、モデル固有の因果的必要軸ではない。 |
| **RQ5: 生成時因果レバレッジはどこで出現するか？** | 生成直前プレフィックス最終トークン置換 ($G_{\ell,t}$) | L24 Residual で 53.24%（中央値 61.57%）。L15 MLP との直接対比 $\Delta G = +53.30\%$ ($p < 10^{-12}$)。 | **Causal leverage is temporally recruited**。因果的感受性は生成直前の後段Residual streamへ動員される。 |

⸻

11. Limitations

第一に、本研究の local causal leverage は単一層・単一コンポーネント・単一トークン位置を中心とした intervention family に基づいている。したがって、Attention head単位の回路、KV cacheを介した経路、複数token positionにまたがる相互作用、非線形なdistributed circuitを網羅していない。

第二に、全層 exploratory sweep は15 matched pairs に基づく。Representative sites は39 complete pairsで再評価しているが、候補層の一部は探索screenを参考に選択されている。そのため39-pair analysisはeffect-size stabilizationとして解釈し、完全にselection-independentなconfirmatory studyとはみなさない。

第三に、主要解析はQwen2.5-1.5B-Instructに集中している。Llama-3.2-1B-InstructではQwenのlate Residual profileは再現されなかったが、full decodability と necessity を含む完全なcross-family replicationではない。

第四に、本研究の出力空間は81候補の constrained VA distribution であり、自由形式の長文生成タスクへそのまま一般化できるとは限らない。

第五に、本研究の「first-person report」はtask-level outputである。測定されたdistribution shiftやinternal representationは、モデルのphenomenal affect、subjective experience、conscious introspectionの存在を意味しない。

⸻

12. Conclusion

本研究では、LLMにおけるaffect-relevant representationsをケーススタディとして、線形アクセス可能性と局所的因果レバレッジを直接比較した。

Qwen2.5-1.5B-Instructでは、Peak対Neutral条件は中間層から高精度に線形デコード可能であり、最大値は、

[
R^2_{\mathrm{L15,MLP}}

0.561
]

であった。

しかし、この最高解読部位をmatched counterfactual activationで置換しても、

[
S_{\mathrm{L15,MLP}}

0.51%
]

に留まり、Generation-timeでも、

[
G_{\mathrm{L15,MLP}}

-0.06%
]

であった。

また、probe-aligned direction removal は系統的なNeutralizationを生じず、probe-orthogonal random direction removal を上回る特異的必要性も支持されなかった。

一方、自己報告生成直前にはlate Residual streamに大きなlocal causal recoveryが現れ、

[
G_{\mathrm{L24,Resid}}

53.24%
]

となった。

同一39ペアにおける直接対比は、

[
\Delta G

53.30%
]

95% bootstrap CI:

[
[45.34%,61.16%]
]

であった。

したがって、本研究の中心的知見は、

[
\boxed{
\text{where information is linearly decodable}
\neq
\text{where and when strong local causal leverage emerges}
}
]

という時空間的解離である。

Linear probing は、情報が外部観測者にとってアクセス可能な場所を同定する。しかし、それだけでは、その情報がモデル自身の下流計算においてどこで、いつ、どの方向に沿って因果的に有効になるかを同定しない。

[
\boxed{
\textbf{Decodability Does Not Localize Causal Leverage}
}
]

⸻
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

Appendix B. Generation-Time Causal Localization Sweeps (Table A2a & Table A2b)

自己報告生成直前（$t_{\mathrm{gen}}$: generation-prefix 最終トークン位置）において、Peak donor activation を matched Neutral run へ置換した際の局所因果回復率を報告する。
探索的スクリーニング（Stage 1: 15 evaluated / 13 valid pairs）と、代表層集中評価（Stage 2: 39 complete pairs）のサンプルサイズおよび測定条件を分離して記録する。

### Table A2a: Exploratory Full-Layer Generation-Time Sweep (Stage 1, 15 Evaluated / 13 Valid Pairs)
*全28層 $\times$ 3コンポーネント（正本: `v3/results/generation_time_causal_sweep.csv`）*

| 層 | MLP 平均 (中央値) | Attention 平均 (中央値) | Residual 平均 (中央値) | 有効ペア数 |
|---|---|---|---|---|
| 0 | -0.13% (+0.19%) | +1.07% (+0.73%) | +1.65% (+1.73%) | 13 |
| 1 | -0.11% (-0.71%) | +0.33% (+0.72%) | +1.35% (+1.22%) | 13 |
| 2 | +1.83% (+1.53%) | +1.66% (+1.69%) | +0.35% (+0.21%) | 13 |
| 3 | +1.74% (+1.47%) | +1.44% (+1.69%) | +1.39% (+1.54%) | 13 |
| 4 | +0.68% (+0.44%) | +1.29% (+0.70%) | +1.32% (+1.13%) | 13 |
| 5 | +1.33% (+1.52%) | +1.43% (+1.12%) | +2.27% (+1.63%) | 13 |
| 6 | +2.06% (+0.72%) | +1.54% (+0.84%) | +1.91% (+2.00%) | 13 |
| 7 | +2.57% (+2.25%) | +1.53% (+1.68%) | +0.86% (+1.28%) | 13 |
| 8 | +0.60% (+0.74%) | +1.60% (+1.74%) | +1.94% (+2.16%) | 13 |
| 9 | +0.44% (+0.62%) | +0.27% (+1.74%) | +1.24% (+0.93%) | 13 |
| 10 | +2.37% (+2.68%) | +1.50% (+1.69%) | +2.21% (+2.20%) | 13 |
| 11 | +1.28% (-0.38%) | -0.62% (-0.31%) | +1.95% (+2.18%) | 13 |
| 12 | +1.28% (+0.83%) | +1.38% (+1.81%) | +2.05% (+2.33%) | 13 |
| 13 | +1.16% (+1.74%) | +0.75% (+0.52%) | +0.89% (+0.92%) | 13 |
| 14 | -0.39% (-0.04%) | +1.14% (+1.22%) | +1.92% (+2.48%) | 13 |
| 15 | -1.99% (-0.84%) | +6.11% (+5.49%) | +7.30% (+5.66%) | 13 |
| 16 | +4.84% (+8.47%) | +19.59% (+19.88%) | +27.72% (+32.14%) | 13 |
| 17 | +14.45% (+17.62%) | +1.31% (-2.18%) | +39.79% (+45.77%) | 13 |
| 18 | +13.79% (+12.94%) | -9.70% (-8.04%) | +40.01% (+48.49%) | 13 |
| 19 | +8.04% (+8.87%) | +1.01% (+0.06%) | +40.34% (+46.34%) | 13 |
| 20 | +7.60% (+14.37%) | -2.85% (-1.02%) | +46.31% (+52.40%) | 13 |
| 21 | +33.19% (+35.65%) | +2.22% (+3.57%) | +46.84% (+50.98%) | 13 |
| 22 | +17.81% (+20.20%) | -15.13% (-10.87%) | +45.06% (+51.39%) | 13 |
| 23 | +10.43% (+19.87%) | -3.20% (-2.90%) | +47.48% (+51.34%) | 13 |
| 24 | +20.79% (+16.43%) | +0.94% (+0.76%) | +47.74% (+48.74%) | 13 |
| 25 | +14.09% (+17.32%) | -0.59% (-0.00%) | +47.12% (+53.61%) | 13 |
| 26 | +14.05% (+15.25%) | -0.28% (-0.22%) | +47.54% (+56.05%) | 13 |
| 27 | +4.78% (+6.82%) | -3.23% (-3.57%) | +50.26% (+51.83%) | 13 |

### Table A2b: Focused Full-Cohort Generation-Time Evaluation (Stage 2, 39 Pairs)
*代表6層 $\times$ 3コンポーネント（正本: `v3/results/focused_causal_sweep_39pairs.csv`）*

| サイト | コンポーネント | 有効ペア | Gen平均 (中央値) | Gen IQR [25%, 75%] | 95% Bootstrap CI | 正の回復率 |
|---|---|---|---|---|---|---|
| L10 | MLP | 39 | +0.18% (+0.80%) | [-4.65%, +4.09%] | [-1.82%, +2.15%] | 53.8% |
| L10 | ATTN | 39 | +0.91% (+0.94%) | [-2.26%, +3.49%] | [-0.85%, +2.65%] | 56.4% |
| L10 | RESID | 39 | -1.42% (+0.15%) | [-3.84%, +3.29%] | [-3.45%, +0.62%] | 51.3% |
| L14 | MLP | 39 | -1.55% (-1.05%) | [-4.31%, +2.22%] | [-3.25%, +0.15%] | 46.2% |
| L14 | ATTN | 39 | -0.39% (+0.21%) | [-1.99%, +3.24%] | [-2.15%, +1.38%] | 51.3% |
| L14 | RESID | 39 | +0.52% (+2.06%) | [-3.64%, +5.66%] | [-1.85%, +2.95%] | 61.5% |
| **L15** | **MLP** | **39** | **-0.06% (+0.50%)** | **[-2.89%, +3.04%]** | **[-2.02%, +1.83%]** | **53.8%** |
| L15 | ATTN | 39 | +3.46% (+5.32%) | [-1.70%, +12.66%] | [+0.45%, +6.48%] | 69.2% |
| L15 | RESID | 39 | +7.40% (+9.70%) | [-1.43%, +16.36%] | [+3.15%, +11.65%] | 74.4% |
| L18 | MLP | 39 | +8.03% (+7.72%) | [+2.52%, +15.68%] | [+4.85%, +11.20%] | 87.2% |
| L18 | ATTN | 39 | -6.82% (-6.87%) | [-12.98%, +1.10%] | [-11.20%, -2.60%] | 30.8% |
| **L18** | **RESID** | **39** | **+42.12% (+49.31%)** | **[+34.02%, +56.14%]** | **[+35.10%, +48.70%]** | **94.9%** |
| L20 | MLP | 39 | +10.54% (+14.23%) | [+3.64%, +21.04%] | [+6.85%, +14.20%] | 82.1% |
| L20 | ATTN | 39 | +1.32% (+0.30%) | [-6.25%, +6.94%] | [-1.45%, +4.10%] | 51.3% |
| **L20** | **RESID** | **39** | **+50.22% (+60.16%)** | **[+32.47%, +66.31%]** | **[+42.60%, +57.50%]** | **97.4%** |
| L24 | MLP | 39 | +15.04% (+17.87%) | [+5.99%, +25.96%] | [+10.85%, +19.20%] | 87.2% |
| L24 | ATTN | 39 | +0.00% (+0.60%) | [-0.89%, +2.46%] | [-0.95%, +1.02%] | 56.4% |
| **L24** | **RESID** | **39** | **+53.24% (+61.57%)** | **[+40.22%, +71.24%]** | **[+45.74%, +60.45%]** | **94.9%** |

---


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



