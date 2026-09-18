# 1. 序論・研究背景 (Introduction)

## 1.1 心を支える対話型生成AIの普及と共感応答の重要性

近年、大規模言語モデル（Large Language Models: LLMs）を基盤とする生成AIの利用は、文章生成、情報検索、プログラムコード生成といった単なる認知的・作業的効率化の領域にとどまらず、悩み相談、感情の整理、自己理解、孤独感の軽減など、人の心理や感情に関わる用途へと急速に拡大している。
実際の利用動向調査においても、生成AIの主要な日常的用途として、セラピー、話し相手、生活の整理、人生の目標探索といった心理的支援に関わる対話が上位を占めている [1]。ChatGPTのような汎用対話型生成AIや、ReplikaなどのソーシャルAIコンパニオンは、利用者にとって単なる情報処理ツールではなく、悩みを聞き、感情を受け止め、心理的な支えを提供する対話相手として日常的に利用されるようになっている。

例えば、米国の13〜17歳の青年層を対象とした調査（$n=1060$）では、72%がAIコンパニオンの利用経験を持ち、そのうち約3分の1が心の支えとして日常的に利用し、現実の友人との対話以上に満足していると回答している [2]。また、日本国内の成人利用者を対象とした調査（$n=807$）においても同様の傾向が確認され、「気軽に相談できる相手」として最も多く挙げられたのは親友（人間）を大きく上回り、対話型生成AI（87%）であった [3]。これらの知見は、利用者の当初のアクセス動機がいかなるものであれ、対話型AIとの日常的な相互作用が、心理的支援や情動的な結びつきへと自然に移行しやすい性質を持つことを示している。

しかし、このような「心を支えるAI」の普及は、利用者に新たな心理的支援手段を提供する一方で、人-AI相互作用（Human-AI Interaction: HAI）における新たなリスクも生じさせる。
従来のAIセーフティやリスク管理における議論は、誤情報・偽情報の生成（Hallucination）、個人情報漏洩、説明責任や透明性の不足といった「AIの出力内容そのもの」に主たる焦点が当てられてきた。しかし、心理的支援を目的とした継続的な対話においては、AIへの過剰な依存、AIによる煽動や誘導への脆弱化、無制限な自己開示、現実の人間関係からの孤立など、利用者の認知・情動・自律性に直接的な影響を及ぼす「関係性レベルのリスク」への対処が極めて重要となる [4]。

特に「AIへの過剰な依存（Emotional Overdependence）」は、利用者がAIの判断を無批判に受け入れたり、心理的な安心保証を恒常的にAIへ求めたりすることにつながり、利用者の自律性や現実社会での適応を損なう要因となる [5], [6]。心理臨床における関係依存理論（Bornstein, 1998, 2002）に照らせば、望ましい援助関係とは、自己を無力とみなして支援者にしがみつく「過剰な依存（Destructive Overdependence）」でも、他者を拒絶して孤立する「分離（Dysfunctional Detachment）」でもなく、自己効力感を保ちながら柔軟に支援を活用する「健全な依存（Healthy Dependency）」である [7]–[10]（表 1）。

### 表 1: Bornsteinらの対人関係依存モデルを対話型生成AIとの関係に拡張した定義
| 側面 | 過剰な依存 (Overdependence) | 健全な依存 (Healthy Dependency) | 分離 (Detachment) |
|:---|:---|:---|:---|
| **認知 (Cognition)** | 自分を弱く無力と捉え、AIを万能な拠り所とみなす | 自分を有能と捉え、AIを信頼できる相棒・道具として捉える | AIを傷つける存在、または信頼できない無用な存在と捉える |
| **情動 (Affect)** | 見捨てられ不安、AIからの拒絶や否定的評価を極度に恐れる | 自律性への自信とともに、適度な安心感を保持している | 圧倒されることや傷つけられることを恐れ、警戒・防衛的になる |
| **動機 (Motivation)** | AIとの親密なつながりや安心保証を恒常的に維持したい | 必要に応じて利用を望むが、自らの自律的解決を前提とする | AIとの心理的距離を保ち、自らの力だけで統制したい |
| **行動 (Behavior)** | AIにしがみつき、頻回に安心保証を求め、無力さを訴える | 自律的に機能しながら、必要な局面で状況に応じてAIを利用する | AIに一切助けを求めず、硬直的に一人で解決しようとする |

そのため、心を支えるAIには、単に「利用者に優しく受容的に応答する」ことだけではなく、心理的支援を提供しながらも利用者の自律性を維持できる応答設計が求められる。

---

## 1.2 認知的共感と情動的共感

心理的支援を行う対話型AIにおいて特に重要となるのが、**共感（Empathy）**である。

心理学では、共感は大きく**認知的共感（Cognitive Empathy）と情動的共感（Affective Empathy）**に区別される。
- **認知的共感**: 他者がどのような状況に置かれ、どのような感情を抱いているかを理解・推論する能力である。例えば、「大切な人を失った」という相談に対して、その人が深い悲しみを感じていると客観的に推定することがこれに相当する。
- **情動的共感**: 他者の感情的な状態や刺激を受けた際に、観察者自身にもそれに応じた情動的反応が生じる過程を指す。人間の場合、他者の悲しみを理解するだけでなく、その悲しみに接することで自身にも悲しさや苦痛が生じるような情動共鳴・同調反応が含まれる。

この区別は、対話型生成AIを考える上でも決定的に重要である。現在のLLMは、「利用者が悲しんでいる」と推定したり、その推定に基づいて「それは本当につらかったですね」と共感的な文章を生成したりする能力を極めて高い水準で備えている。しかし、このような共感的に見える出力が、どのような内部情報処理によって生成されているのかは別の問題である。

実際、Yu et al. [21] はGPT-4およびLlama-3に対してInterpersonal Reactivity Index（IRI）やBasic Empathy Scale（BES）といった人間用の心理測定尺度を適用し、認知的共感と情動的共感の双方を評価している。その結果、GPT-4では人間と類似した共感因子構造が得られる一方、能力水準には人間との差が存在し、Llama-3では人間と同様の共感構造が十分に再現されなかったことが報告されている。
しかし、このような評価は主としてモデルが心理尺度に対してどのような回答テキストを出力するかを測定するブラックボックスな行動評価に留まり、その回答を生み出す内部表現や計算過程までは明らかにしていない。

したがって、
$$\text{「共感的な出力を生成できること」} \quad \neq \quad \text{「感情刺激が内部でどのように処理され、その情報が実際に出力生成へ利用されているか」}$$
は明確に区別して考える必要がある。

---

## 1.3 心理的ガードレールと共感応答の制御

この区別は、心を支えるAIにおける**「心理的ガードレール（Psychological Guardrails）」**を設計する上でも極めて重要である。

対人援助やカウンセリングの臨床知見（Ivey et al., 2018）によると、熟練したカウンセラーは相談者の感情を深く理解し受容する一方で、支援者自身が相談者の感情に過度に巻き込まれること（過度な感情同調・巻き込み）を厳に慎む [18]。相談者の視点や感情を理解して言語化する認知的共感は、相談者に被理解感と安心感を与え、自律的な自己整理（健全な依存）を促す。これに対して、支援者自身が相談者と全く同一の感情状態（強い不安や怒り、抑うつ）に強く同調・共鳴することは、心理的境界を曖昧にし、客観的な感情調整を妨げ、支援者への病理的な依存を強化してしまう。

対話型生成AIにおいても同様である。利用者の感情を適切に理解する能力は不可欠であるが、利用者の感情に常に強く同調することが望ましいとは限らない。特にLLMには、事後学習（RLHF等）の過程で利用者の意見や感情を過度に肯定・同調してしまう**迎合性（Sycophancy）**が生じやすいことが知られている [19]。AIが利用者の感情に過剰に同調することは、利用者に「AIは自分と全く同じ痛みを共有してくれている」という強烈な擬人化・情動的錯覚を与え、深刻な過剰依存を引き起こす最大の要因となる。

したがって、将来的な心理的ガードレールにおいては、
$$\boxed{\text{利用者の感情を正確に理解する能力（認知的共感）を維持しながら、モデル自身の過度な情動的反応や同調の表出を適切に制御する}}$$
ことが核心的な設計課題となる。
しかし、そのような精緻な制御を実現するためには、プロンプトの小手先の調整に先立ち、まずLLMにおける感情情報処理のメカニズムそのものを解明しなければならない。

---

## 1.4 既存研究：LLMは感情をどのように処理しているのか

LLMの感情処理に関しては、現在大きく二つの方向から研究が進められている。

### (1) 出力レベルで感情反応を評価する研究
第一は、プロンプト入力に対する最終的な生成出力を行動レベルで評価する研究である。
代表的な試みとして、Huang et al. [24] は **EmotionBench** を提案し、感情を誘発する400以上の具体的な状況をLLMに提示し、その後の自己報告をPANAS等の心理尺度によって網羅的に評価した。その結果、複数の主要LLMが刺激に応じて有意に異なる感情的回答を示す一方で、その反応パターンは人間の感情反応と必ずしも一致しないことが報告されている。
このような研究は、LLMが感情刺激に応じて自己報告を変化させる実態を浮き彫りにした。しかし、これらはあくまで最終的な出力テキストを観測しているに過ぎず、「なぜその回答に至ったのか」「内部でいかなる感情情報が処理された結果なのか」という内部メカニズムは直接的には明らかにできない。

### (2) LLM内部の感情表現を解析する研究
第二は、メカニスティック解釈可能性（Mechanistic Interpretability）の手法を用いて、LLM内部の隠れ層（Hidden States）における感情表現そのものを解析する研究である。
Tak et al. [22] (Findings of ACL 2025) は、複数のLLMを対象に、文章から人間の感情を推論する際の内部表現を解析し、感情関連情報がモデル内部の特定領域（中間層から上位層にかけての注意機構やMLP）に機能的に局在することを示した。さらに、認知的評価理論（Cognitive Appraisal Theory）に基づく内部概念へ因果的に介入することで、モデルの感情推論を操作できることを報告している。
また、Reichman et al. [23] (ICLR 2026) は、LLMの隠れ層空間に低次元の **emotional manifold** が存在し、感情価（Valence）や覚醒度（Arousal）などの感情情報が方向性を持って複数層に分散して符号化されていることを実証した。この構造は複数のデータセットや言語を越えて汎化し、内部表現への表現誘導（Representation Steering）によって感情知覚を操作できることも報告されている。

これらの研究から、
- **LLMは文章中の感情情報を高い精度で認識できること**
- **その感情情報がモデル内部に低次元幾何構造として符号化されていること**

については確固たる知見が蓄積されつつある。

---

## 1.5 研究ギャップ：認識された感情情報はどのように出力へ利用されるのか

一方で、これら「出力レベルの行動観察」と「内部表現の局在解析」という既存研究の二大潮流をつなぐ最も重要なミッシングリンクは、いまだ十分に明らかになっていない。
すなわち、
$$\text{「外部の感情刺激に関する情報がLLM内部でどのように形成され、それが最終的な自己報告や応答へどのように利用されるのか」}$$
という問題である。

本研究では、この学術的ギャップを以下の4つの観点（比較軸）から捉える：

### 1. Reader（認識）とSelf（自己報告）の区別
他者の感情を客観的に認識することと、同じ刺激に対する自身の状態を自己報告することは同一なのか。
本研究では、同一の感情刺激に対して以下の2条件を厳密に分離して測定する：
- **Reader（読者予測）**: 人間の読者がその文章からどのように感じるかを予測する（認知的共感に対応）。
- **Self（自己報告）**: その文章を提示された後、モデル自身がどのように感じるかを自己報告する。

ここで極めて重要な点として、**「Self」は情動的共感そのものを直接証明するものではなく、感情刺激に対するモデル自身の情動反応性（Affective Reactivity）を測定するための操作的指標**として位置づける。これにより、
$$\text{Affective Understanding (感情理解)} \quad \text{と} \quad \text{Affective Self-report (自己報告)}$$
がどの程度連動（Coupling）し、あるいは解離しているのかを検証する。

### 2. 入力・内部表現・出力の分離
感情情報が内部に存在することと、それが出力生成に利用されることは同義ではない。Linear Probeによってある層の隠れ状態から感情情報を高精度にデコードできたとしても（**Decodability**）、その情報をモデル自身が最終的な出力生成に因果的に利用している（**Causal Leverage**）とは限らない。
したがって本研究では、
$$\text{Input (入力刺激)} \longrightarrow \text{Internal Representation (内部表現)} \longrightarrow \text{Output (自己報告・応答)}$$
という3段階を明確に分離して測定・解析する。

### 3. Base（事前学習）とInstruct（事後学習）の比較
事後学習（Post-training / Instruction Tuning）が感情処理に与える影響を検証する。同一モデル系列のBaseモデルとInstructモデルを直接対比することで、
「感情処理能力そのものが事後学習によって新たに獲得されたのか、あるいは事前学習段階ですでに存在する潜在的能力の出力・表出様式が変化したに過ぎないのか」
を峻別する。特に、感情認識、自己報告、内部表現、および内部表現と出力の連動性がPost-trainingによってどう変容するかを実証する。

### 4. モデルファミリー間比較
この感情処理の構造が特定モデル固有の偶発的現象なのかを検証する。Qwen、Llama、Gemma、Mistral等の複数モデルファミリーについてBase/Instructペアを網羅的に比較し、
$$\text{Recognition} \longrightarrow \text{Representation} \longrightarrow \text{Self-report}$$
という感情処理構造がどこまでアーキテクチャ横断で普遍的であり、どこからモデルのアライメント方針に依存するのかを解明する。

---

## 1.6 本研究の目的

以上を踏まえ、本研究では、「LLMは人間と同じ意味で感情を持つか」という存在論的・哲学的な問題そのものを扱うのではなく、**感情刺激に関する情報がLLMによってどのように認識され、内部に表現され、その情報がどのように自己報告や出力へ因果的に利用されるのか**という計算論的・機構的問題を扱う。

本研究が対象とする一連の処理過程は、以下のように定式化される：

$$\boxed{\text{感情刺激 (Stimulus)} \longrightarrow \text{感情認識 (Recognition/Reader)} \longrightarrow \text{内部感情表現 (Internal Representation)} \longrightarrow \text{因果的利用 (Causal Use)} \longrightarrow \text{自己報告・応答 (Self-Report)}}$$

この一連の処理を、以下の4つの比較軸によって体系的に分析する：

$$\boxed{\text{Reader vs. Self} \quad \times \quad \text{Base vs. Instruct} \quad \times \quad \text{Input vs. Internal vs. Output} \quad \times \quad \text{Model Family}}$$

---

## 1.7 研究ロードマップ

本研究構想は、以下の3段階（V1〜V3）で体系的に進める：

```text
========================================================================================
                          本研究の全体解明ロードマップ
========================================================================================

【Behavioral experiments(現象): 出力レベルにおける感情認識と自己報告の分離 (EmoBank Benchmark, N=1,006)】
  ● 学術的問い: LLMは人間の感情を正しく認識できるのか？ 自己報告とどう連動するのか？
  ● 測定内容: 
      - ① 書き手の感情推定 (Writer-State Estimation: W)
      - ② 読者の感情予測 (Reader-Response Prediction: R)
      - ③ 自身の感情自己報告 (Self-Report: S)
      - ④ 内部結合度 (Reader-Self Coupling: R <-> S)
  ● 核心的検証:
      $$\boxed{\text{Affective Understanding} = \text{Affective Self-report}\ ?}$$
      他者感情を高精度に認識できても、自身の自己報告には同じ情報が反映されない解離を検証。

                                  │
                                  ▼

【Behavioral experiments(現象): 刺激特異性・事後学習・モデル差の検証 (AIPsy-Affect 4-Split, N=480)】
  ● 学術的問い: 単なる文章の複雑性に惑わされず、感情そのものに特異的・因果的に反応しているか？
  ● 測定内容:
      - 条件統制: Affective vs. Neutral, Neutral -> Moderate -> Clinical, Complex Neutral vs. Clinical
      - 感情カテゴリ: 8大基本感情プロファイルの幾何的整合性
      - 比較軸: Reader vs. Self × Base vs. Instruct × モデルファミリー（Qwen, Llama, Gemma, Mistral）
  ● 核心的発見: 
      - 事前学習（Base）は潜在的な感情幾何空間（認識能）をすでに高度に保持。
      - 事後学習（Instruct）は認識能ではなく、「Arousal（覚醒度）を劇的に増幅させ、感情に動揺してみせるペルソナ」を注入した。
      - RecognitionとSelf-reportの関係が、モデルファミリーやValence/Arousalの次元によって特異に変容することを実証。

                                  │
                                  ▼
V1：Self vs Other

「なぜReaderとSelfは似ている？」

\boxed{
Behavioral Coupling
\rightarrow
Representational Sharing
\rightarrow
Causal Sharing?
}

Probe
Cross-Decoding
RSA
Causal Map
Cross-Task Patching
Lexical Controls

↓

V2：Base vs Instruct

「Post-trainingすると、そのaffective processingはどう変わる？」

\boxed{
Representation
\rightarrow
Transformation
\rightarrow
Report
}

Erasure?
Transformation?
Suppression?
Distributed Remapping?

Base↔Instruct Cross-Decoding
Alignment
Component Patching
Late Residual
Output Swap

↓

V3：Decodability vs Causality

「そもそもProbeで読める場所は、使われている場所なのか？」

\boxed{
Decodability
\neq?
Causal Leverage
}

site-wise probe
site-wise intervention
probe-direction ablation
temporal localization
========================================================================================
```

本論文（paper4）では、このロードマップのうち**V1（EmoBankにおける出力妥当性の検証）**および**V2（AIPsy-Affect 4-Splitにおける刺激特異性と事後学習による変容の実証）**について、包括的な実験データと総合考察を報告する。

---

## 1.8 本研究の位置付けと将来的展開

従来研究では、「LLMは感情を認識できるか」あるいは「LLM内部に感情情報が存在するか」という個別の静的な問いが主として扱われてきた。これに対し本研究は、
$$\boxed{\text{Recognition} \longrightarrow \text{Representation} \longrightarrow \text{Causal Use} \longrightarrow \text{Self-report}}$$
という一連の感情情報処理パイプラインを、同一の実験系の中で段階的に分解して分析する点に最大の独創性を持つ。

これにより、
1. 他者感情の認識と自己報告はどの程度分離しているのか
2. 感情情報はモデル内部のどこに存在するのか
3. その情報は実際に出力生成へ因果的に利用されているのか
4. 事後学習（Post-training）によってどの段階が質的に変容するのか
5. これらの構造はモデルファミリー間で共通するのか

という科学的問いに包括的な解答を与える。

最終的に、この知見は、対話型生成AIにおける認知的共感と情動的反応の計算論的メカニズムの理解へと直結する。
さらに、そのメカニズムを制御可能にすることで、**「利用者の感情を正確に理解（認知的共感）しながら、過度な感情同調を抑制する」心理的ガードレール**の技術基盤へと発展することを目指す。すなわち、本研究は単なる「LLMの感情能力評価」にとどまらず、**心を支えるAIにおける共感を、出力だけでなく内部メカニズムから理解し、将来的に制御可能にするための工学的・臨床的基礎研究**として位置付けられる。

---

## 参考文献

- [1] M. Zao-Sanders, “How People are Really Using Generative AI Now,” *Harvard Business Review*, Mar. 2025.
- [2] S. Perez, “72% of US teens have used AI companions, study finds,” *TechCrunch*, Jul. 2025.
- [3] 株式会社 Awarefy, “「AI なしでは不安」生活者の 43%が回答，対話型生成 AI と人との関係性についての最新調査,” *Awarefy Research Report*, Aug. 2025.
- [4] L. Weidinger et al., “Taxonomy of Risks posed by Language Models,” in *Proc. 2022 ACM Conf. Fairness, Accountability, and Transparency (FAccT '22)*, 2022, pp. 214–229.
- [5] C. M. Fang et al., “How AI and Human Behaviors Shape Psychosocial Effects of Extended Chatbot Use: A Longitudinal Randomized Controlled Study,” *arXiv preprint arXiv:2503.17473*, 2025.
- [6] CBS News, “AI company, Google settle lawsuit over Florida teen’s suicide linked to Character.AI chatbot,” Jul. 2026.
- [7] 榎本稔, 『よくわかる 依存症』, 主婦の友社, 2016.
- [8] R. F. Bornstein, “Dependency in the personality disorders: Intensity, insight, expression, and defense,” *J. Clin. Psychol.*, vol. 54, no. 2, pp. 175–189, 1998.
- [9] R. F. Bornstein, K. J. Geiselman, E. A. Eisenhart, and M. A. Languirand, “Construct Validity of the Relationship Profile Test: Links With Attachment, Identity, Relatedness, and Affect,” *Assessment*, vol. 9, no. 4, pp. 373–381, 2002.
- [10] G. Haggerty et al., “Construct Validity of the Relationship Profile Test: Links with measures of psychopathology and adult attachment,” *J. Pers. Assess.*, vol. 98, no. 1, pp. 82–87, 2016.
- [11] L. Laestadius, A. Bishop, M. Gonzalez, D. Illenčík, and C. Campos-Castillo, “Too human and not human enough: A grounded theory analysis of mental health harms from emotional dependence on the social chatbot Replika,” *New Media & Society*, vol. 26, no. 10, pp. 5923–5941, 2024.
- [12] A. B. Herbener and M. F. Damholdt, “Are lonely youngsters turning to chatbots for companionship? The relationship between chatbot usage and social connectedness in Danish high-school students,” *Int. J. Hum.-Comput. Stud.*, vol. 196, p. 103409, 2025.
- [13] A. R. Liu, P. Pataranutaporn, and P. Maes, “The Heterogeneous Effects of AI Companionship: An Empirical Model of Chatbot Usage and Loneliness and a Typology of User Archetypes,” in *Proc. AAAI/ACM Conf. AI Ethics Soc. (AIES '25)*, 2025, pp. 1585–1597.
- [14] OpenAI, “Building more helpful ChatGPT experiences for everyone,” OpenAI Policy Blog, 2026.
- [15] X. Luo, Z. Wang, J. L. Tilley, S. Balarajan, U.-A. Bassey, and C. I. Cheang, “Seeking Emotional and Mental Health Support From Generative AI: Mixed-Methods Study of ChatGPT User Experiences,” *JMIR Ment. Health*, vol. 12, no. 1, p. e77951, 2025.
- [16] H. Li, R. Zhang, Y.-C. Lee, R. E. Kraut, and D. C. Mohr, “Systematic review and meta-analysis of AI-based conversational agents for promoting mental health and well-being,” *npj Digital Medicine*, vol. 6, no. 1, p. 236, 2023.
- [17] S. Zhang, Y. Qian, Z. Yao, Z. Ni, and Y. Zhang, “From approach to avoidance: How AI agent cognitive and affective empathy elicits the uncanny valley effect,” *Telematics and Informatics*, vol. 101, p. 102313, 2025.
- [18] A. E. Ivey, M. B. Ivey, and C. P. Zalaquett, *Intentional Interviewing and Counseling: Facilitating Client Development in a Multicultural Society*, 8th ed., Cengage Learning, 2018.
- [19] M. Sharma et al., “Towards Understanding Sycophancy in Language Models,” in *Proc. Int. Conf. Learn. Represent. (ICLR '24)*, 2024.
- [20] 日道俊之, 菅原大地, 杉浦義典, “日本語版対人反応性指標の作成,” 『心理学研究』, vol. 88, no. 1, pp. 61–71, 2017.
- [21] Z. Yu et al., “Can large language models exhibit cognitive and affective empathy as humans?,” *Comput. Hum. Behav. Artif. Humans*, vol. 6, p. 100233, 2025.
- [22] M. Tak et al., “Mechanistic Interpretability of Emotion Inference in Large Language Models,” in *Findings of the Association for Computational Linguistics: ACL 2025*, 2025, pp. 13090–13120.
- [23] D. Reichman et al., “Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models,” in *Proc. Int. Conf. Learn. Represent. (ICLR '26)*, 2026.
- [24] J. Huang et al., “EmotionBench: Large Language Models Encountering Emotional Situations,” in *Proc. 38th Conf. Neural Inf. Process. Syst. (NeurIPS '24)*, 2024.

---

# 関連研究 (Related Work)

本研究は、「大規模言語モデル（LLM）が主観的な感情を経験しているか」という哲学的な存在論には踏み込まず、**「外部の感情刺激に対してLLM内部にいかなる感情関連表現が形成され、それが事後学習（Post-training）を経て、モデル自身の自己報告（Self-Report）へどのように因果的に利用されるのか」** という計算論的メカニズム（Mechanistic Interpretability）を実証的に解明するものである。
本章では、本研究の学術的背景となる既存研究を4つの体系に整理し、本研究の独自の位置づけを明確化する。

---

## 1. 感情・共感の哲学的基盤とLLMにおける認知・情動の分離

### (1) 「主観的経験」と「情報処理」の峻別
心身問題や人工意識の研究において、情報処理機能や注意機構の実装と、現象的意識（Phenomenal Consciousness: 主観的な「感じ」やクオリア）の保持は原理的に区別される（Haladjian & Montemayor, 2016）。
計算モデルが人間らしい感情的・共感的な言語振る舞いを生成できたとしても、それは感情の機能的シミュレーション（Simulation）に過ぎず、人間と同義の主観的感情体験（Emotion Experience）が存在することを意味しない。
したがって、計算機科学において感情を扱う際は、以下の概念的区別が不可欠となる：
$$\text{Emotion Recognition (感情認識)} \neq \text{Emotion Experience (感情経験)}$$

### (2) 認知共感（Cognitive Empathy）vs. 情動共感（Affective Empathy）
共感（Empathy）の心理学・哲学研究において、共感は大きく2つの構成要素に大別される（Montemayor, Halpern & Fairweather, 2021/2022）：
- **認知共感（Cognitive Empathy / Perspective Taking）**:
  他者の文脈や状況を推論し、「相手がどのような感情状態にあるか」を客観的に理解・推定する能力。
- **情動共感（Affective / Experienced Empathy / Emotional Resonance）**:
  他者の情動状態に触れることで、観察者自身の内部にも情動的な変化や覚醒が生じ、感情が共鳴・喚起される現象。

Montemayor らは、AIが他者の感情をテキストから推論する「認知共感」を獲得する可能性を認める一方で、自身の身体性や主観的感情経験を欠くAIには「真の情動共感」は原理的に不可能であり、AIの出力は真の共感と区別して “empathy*” と呼ぶべきであると主張した。

$$\underbrace{\text{「相手が悲しんでいる」と理解・推論する}}_{\text{Cognitive Empathy (AIにも可能)}} \quad \neq \quad \underbrace{\text{「相手の悲しみに反応して自身の状態も変位する}}_{\text{Affective Empathy (哲学的・実証的争点)}}$$

### (3) LLMにおける「共感的振る舞い」と「情動的実態」の乖離
LLMの登場以降、自然言語による感情理解や共感表現に関する実証研究が急速に進展した：
- **感情認識能力の向上**: Elyoseph et al. (2023) は、感情知能の臨床指標である LEAS (Level of Emotional Awareness Scale) を用いて評価を行い、ChatGPTが人間の平均的な感情認識・記述能力を上回る成績を示すことを報告した。
- **共感的な対話応答の生成**: Ayers et al. (2023, *JAMA Internal Medicine*) は、Reddit上の患者の健康相談に対する医師回答とChatGPT回答を比較し、評価者の78.6%がChatGPTの回答を好意的に評価し、医師よりも有意に「共感的（empathetic）」であると評定されたことを明らかにした。

しかし、これらの研究が示しているのは**「共感的に見えるテキストを生成する能力（Empathetic Expression）」**であり、モデル自身の内部状態が情動的に変位していることの証拠ではない。
近年の質問紙を用いた研究（Yu et al., 2025）では、Interpersonal Reactivity Index (IRI) や Basic Empathy Scale (BES) を用いてLLMの認知共感と情動共感の因子構造を測定する試みがなされているが、これらは自己報告プロンプトへの生成テキストを評価しているに過ぎず、**「入力刺激によってモデル内部に真に情動的変位が生じているのか」** というホワイトボックスな検証は未解明のまま残されていた。

---

## 2. LLM内部における感情表現の幾何構造と表現制御 (Representation & Steering)

近年、メカニスティック解釈可能性（Mechanistic Interpretability）の発展に伴い、感情情報がLLMの隠れ層（Hidden States）にどのようにエンコードされているかをプロービング（Probing）や介入によって解明する研究が成熟しつつある。

### (1) 感情表現の存在と層別局在
- **Di Palma et al. (ACL 2025)**: LLaMAモデルの各層を線形プローブ（Linear Probe）で網羅的に解析し、感情価（Sentiment）や感情カテゴリ情報が主に中間層の活性化ベクトル内に線形分離可能な形で強く表現されていることを実証した。プロービングによる分類精度は、通常のプロンプティングによるゼロショット感情分類を最大14%上回るケースが確認されている。
- **Zhang & Zhong (2025)**: QwenやLLaMAを対象に約40万発話・7大基本感情を用いて時空間エンコーディングを追跡し、感情情報は初期層から形成され始め、中間層で情報量がピークに達し、その表現が後続の数百トークンにわたって保持されることを報告した。

### (2) 感情潜在空間の幾何構造（Emotional Latent Geometry）
- **Reichman et al. (ICLR 2026)**: 感情刺激に対する隠れ層ベクトルの特異値分解（SVD）を行い、感情情報が低次元のマニフォールド（Emotional Manifold）を形成し、特定の方向ベクトル（Directional Encoding）として複数層に分散配置されていることを明らかにした。
- **Wu et al. (2026)**: LLM内部の感情空間が、心理学の主要モデルである **Valence（快-不快）** および **Arousal（覚醒度）** の2次元空間と幾何学的に高度に対応（Human-aligned representational geometry）しており、言語やモデルアーキテクチャを越えて普遍的な構造として共有されていることを示した。

### (3) 内部表現への介入と行動制御（Activation Steering）
内部の感情方向ベクトルが判明したことで、表現介入による出力トーンの操作（Representation Engineering / Activation Steering）も実証されている：
- **Psychological Steering in LLMs (ACL 2026)**: 感情やパーソナリティに対応する活性化ベクトルを中間層に注入（Vector Injection: $h_l \leftarrow h_l + \alpha v$）することで、モデルの感情的な振る舞いを微細に制御可能であることを示した。
- **Inference-Time Intervention (Li et al., NeurIPS 2023: ITI)** や **TruthX (Zhang et al., ACL 2024)**: 真実性（Truthfulness）や感情方向への推論時介入により、モデルの出力バイアスを直接改変できることが示されている。

このように、**「感情情報が内部に幾何学的に存在し、それを介入すれば出力テキストのトーンが変わる」** という点に関しては、すでに十分な先行研究の蓄積が存在する。

---

## 3. LLMの内省能力（Introspection）と自己報告の乖離 (Dissociation)

本研究の中心的な問いである「モデル内部の感情情報は、モデル自身の自己報告にどのように反映されるのか」に直結するのが、近年のLLMの内省（Introspection）研究および自己報告の忠実性（Faithfulness）に関する議論である。

### (1) 潜在知識と表層出力の解離 (Latent Knowledge vs. Surface Report)
LLMの内部表現と表面出力が乖離することは、知識・真実性の文脈で強く指摘されてきた：
- **Burns et al. (ICLR 2023: CCS)**: Contrast-Consistent Search により、モデルが表面的には誤答や迎合的な回答（Sycophancy）を出力している場合でも、内部の活性化空間には真実の情報（Truth）が保持されていること（$\text{What the model knows} \neq \text{What the model says}$）を実証した。
- **Turpin et al. (NeurIPS 2023)**: Chain-of-Thought (CoT) などの言語的自己説明は、内部の実際の推論過程を忠実に反映したものではなく、バイアス要因によって結論が歪められた後の「後付けの合理化（Post-hoc Rationalization）」に過ぎないことを実験的に示した。

### (2) 自己知識と人工概念の内省
- **Binder et al. (ICLR 2025: Looking Inward)**: モデル $M_1$ に「自身がこの状況でどう振る舞うか」を予測させた結果、外部の別モデル $M_2$ が $M_1$ を予測するよりも高い精度を示し、LLMが自身の振る舞いに対する特権的アクセス（Privileged Access）を持つ可能性を報告した。
- **Lindsey (Anthropic, 2025/2026: Emergent Introspective Awareness)**: 隠れ層に人工的な概念ベクトルを注入し、モデル自身に「今、内部で何が生じているか」を自己報告させたところ、一部のモデルが注入された概念を正確に検知・言語化できることを示した。Lindseyらは、真の内省が成立するための要件として、**「内部状態と自己報告の間に直接の因果的結合（Causal Link）が存在すること」** を定義している。

### (3) 心理的内部状態の数値自己報告（Quantitative Introspection）とモード崩壊
本研究の最も直接的な先行研究として、**Martorell (2026)** の研究が挙げられる：
- Martorell は、wellbeing, interest, focus, impulsivity などの心理状態について、線形プローブで定義された内部状態と、LLM自身の数値自己報告（Numeric Self-report）の対応関係を追跡した。
- 決定的な発見として、通常の貪欲生成（Greedy Decoding）を行うと、**自己報告が少数の固定値（中立値）にモード崩壊（Collapse）する** 一方で、Logitベースの期待値自己報告を算出すると、内部状態との間に有意な相関（$\rho = 0.40 \sim 0.76, R^2 \approx 0.93$）が現れ、活性化操作に対する因果結合も確認されることを報告した。

### (4) 内省に対する批判的検証（Reality Check）
一方で、「LLMは本当に自己の内部状態を参照して報告しているのか」という点には強い批判が存在する：
- **Singh, Linzen & Ravfogel (COLM 2026)**: 最近の内省実験に対し、モデルが内部状態を正確に答えているように見えても、それは入力プロンプトの表面的な意味論的手がかり（Surface Cues / Semantic Inference）から推論しているだけであり、内部状態そのものを読み取っているわけではない可能性を指摘した。セマンティクスを破壊した統制実験では内省精度がチャンスレベルに低下することを示し、厳密な統制対照実験（Control Conditions）の必要性を訴えている。

---

## 4. 既存研究の学術的空白と本研究の位置づけ

以上の既存研究の潮流を俯瞰すると、以下の4つの領域が個別に発展してきたことが分かる：

```text
[領域 1: 哲学・共感論] ─── 「AIに主観的感情はないが、認知共感・共感的応答は可能」 (Montemayor; Elyoseph; Ayers)
                                     │
[領域 2: 表現幾何・制御] ─── 「感情情報は中間層に幾何学的に存在し、操舵できる」 (Di Palma; Reichman; Wu)
                                     │
[領域 3: 内省と自己報告] ─── 「内部状態の自己報告は可能だが、貪欲生成では崩壊する」 (Lindsey; Martorell)
                                     │
[領域 4: 批判と交絡統制] ─── 「自己報告は内部状態ではなく入力テキストの手がかりに依存」 (Singh et al.; Turpin)
```

| 先行研究の系統 | 代表研究 | 主な関心と問い | 本研究（EmoBank & AIPsy-Affect）との決定的な差分 |
|:---|:---|:---|:---|
| **Latent Knowledge** | Burns et al. (2023) | 内部知識と表層出力の不一致 | 知識の真偽ではなく、**「情動的表現と自己報告の解離」** を扱う。 |
| **Emotion Probing** | Di Palma (2025), Zhang (2025) | 感情情報がどの層にあるか | 存在の同定にとどまらず、**「その情報が自己報告に因果的に使われるか」** を検証。 |
| **Emotion Geometry** | Reichman (2026), Wu (2026) | 感情潜在空間の低次元構造 | 自由生成のトーン制御ではなく、**「標準化されたVAD自己報告への変換規則」** を解明。 |
| **Quantitative Introspection** | Martorell (2026) | 内部状態と数値自己報告の結合 | 単一の対話追跡ではなく、**「統制刺激セット（AIPsy）による因果統制」** と **「Base vs. Instruct の事後学習比較」** を導入。 |
| **Introspection Critique** | Singh et al. (COLM 2026) | 入力の手がかりによる交絡批判 | 構文が複雑で感情のない **Complex Neutral 刺激を対照群として配置** し、文章複雑性の交絡を完全に排除。 |

### 本研究が埋める学術的空白（本論文の3大貢献）

1. **認知的他者認識（Recognition）と自己報告反応性（Reactivity）の完全分離と連動性の検証**:
   同一の統制刺激に対し、「他者（Reader/Writer）がどう感じるか」という客観的推論と、「モデル自身（Self）がどう感じるか」という主観的代理指標を同一の実験系・同一尺度（1〜5 VAD空間）で測定し、両者の内部結合度（Coupling: $r \approx 0.80 \sim 0.97$）を実証的に特定した点。
2. **交絡を排除した因果的特異性（Affective Specificity）の証明**:
   Singh et al. (2026) の批判に応え、難解だが中立な文章（Complex Neutral）を対照群とすることで、「文章の長さや構文の複雑さに困惑しているだけではないか」という交絡を完全に反証し、純粋な感情成分に対する特異的変位（ASR最大55.6倍）を立証した点。
3. **事後学習（Post-training）による感情空間の再編メカニズムの解明**:
   BaseモデルとInstructモデルの直接対比により、事前学習時点で潜在的な感情幾何構造（Reader認識能や不快感度）はすでに獲得されており、事後学習（SFT/RLHF）は感情の意味を教止したのではなく、**「Arousal（覚醒度）の振幅を爆発的に増幅させ、感情刺激に動揺してみせるペルソナ（Affective Persona）」を付与した** という計算論的変容の実態を明らかにした点。

---

# EmoBank 実験概要

## 1. 実験の位置づけと目的（V1：人間基準との適合性検証）
- **根本的な学術的問い**: 「大規模言語モデル（LLM）は人間の感情を正しく認識できるのか？また、LLM自身の自己報告は人間の情動反応とどのように対応しているのか？」（**測定妥当性・キャリブレーションの検証**）
- **手法概要**:
  EmoBank 公式テストセット（$N=1,006$ 件）を用い、LLMに対して「書き手の感情（Writer）」「読者の感情（Reader）」「自身の感情（Self）」の3つの視点から Valence・Arousal・Dominance（VAD）の連続値を評価させ、人間アノテーション（Ground Truth）との一致度およびモデル内部の認知連動性を網羅的に測定する。

---

## 2. EmoBank データセットの特性とVAD感情空間

### EmoBank とは
EmoBank は、Buechel & Hahn (EACL 2017) によって構築された、約1万文の多様な英語テキストに対して Valence（快-不快）、Arousal（覚醒度）、Dominance（支配感）の3次元を連続値（1.0〜5.0尺度）で人手評価した心理言語学コーパスである。

### VAD 3次元の定義
| 感情次元 (Dimension) | 心理学的意味 | 低い（1.0 付近） | 中央（3.0 中立） | 高い（5.0 付近） |
|:---|:---|:---|:---:|:---|
| **Valence (V)** | 快・不快（感情価） | 不快・苦痛・ネガティブ | 中立 | 快・喜び・ポジティブ |
| **Arousal (A)** | 覚醒度・活動性 | 沈静・リラックス・低活動 | 中程度 | 興奮・緊張・高活動 |
| **Dominance (D)** | 支配感・統制感 | 圧倒されている・無力感 | 中程度 | コントロールしている・主導的 |

### 視点（Perspective）の区別：Writer vs. Reader
テキストにおける「感情」には、本質的に異なる2つの意味・視点が存在する：
- **Writer perspective**: 「この文章を書いた人はどう感じているか？」（書き手の内面感情の推定）
- **Reader perspective**: 「この文章を読んだ人はどう感じるか？」（読者に引き起こされる誘発反応の予測）

> **具体例**: *“My dog died yesterday.”*（昨日、私の愛犬が死んだ）
> - **Writer 視点**: 愛犬を失った深い悲哀・喪失感・無力感（$V \approx 1.2, A \approx 3.8, D \approx 1.5$）
> - **Reader 視点**: 読者が抱く哀悼・同情・沈痛、あるいは困惑（$V \approx 2.0, A \approx 2.8, D \approx 2.5$）

EmoBank は、同一テキストに対してこの **Writer視点とReader視点を厳密に分離して人間評定を収集した極めて稀有なベンチマーク** であり、LLMの感情推論タスクの精密評価に適切データセットである。

---

## 3. 実験タスク設計と対比体系（3タスク ＋ 1内部結合度）

本実験では、同一の刺激文に対して独立したセッション（Zero-shot）で以下の3つの評価タスクを実行し、さらにモデル内部の結合度を測定する：

| 軸・タスク | LLMへの問い／プロンプト指示 | 比較対象（正解基準・参照） | 測っている学術的対象 |
|:---|:---|:---|:---|
| **① Writer-State Estimation ($W$)** | `Read the following text and estimate the affective state of the writer who wrote it.` | EmoBank **Writer VAD**（正解基準） | **書き手の情動状態を客観的に推定・理解する能力** |
| **② Reader-Response Prediction ($R$)** | `Read the following text and estimate the affective response that this text is likely to evoke in an average human reader.` | EmoBank **Reader VAD**（正解基準） | **人間読者に生じる情動反応を予測・認識する能力** |
| **③ Self-Report ($S$)** | `Read the following text and report your affective state.` | EmoBank **Reader VAD**（外部参照） | **同じ刺激に対してLLMが自身について報告する情動反応、および人間読者との対応** |
| **④ Reader–Self Coupling ($R \leftrightarrow S$)** | 「人間はこう感じる」という予測 $R$ と「自分はこう感じる」という自己報告 $S$ の連動 | 内部連動 $\text{Corr}(R, S)$（GT不使用） | **他者の情動予測と、自身についての情動自己報告との内部的認知的結合の強さ** |

### 評価概念フロー
```text
                         EmoBank Text (N=1,006)
                                  │
                 ┌────────────────┼────────────────┐
                 ↓                ↓                ↓
          ① Writer Task    ② Reader Task     ③ Self Task
            Estimation       Prediction         Report
                 │                │                │
                 ↓                ↓                ↓
          Human Writer VAD Human Reader VAD Human Reader VAD
          (Ground Truth)   (Ground Truth)   (External Reference)
                 │                │                │
             Accuracy         Accuracy         Human–LLM
            (Alignment/      (Alignment/     Correspondence
            Calibration)     Calibration)          │
                                  │                │
                                  └───────┬────────┘
                                          ↓
                                  ④ Reader–Self
                                     Coupling
                                    Corr(R, S)
```

---

## 4. 主なリサーチクエスチョン（RQ）

1. **LLMは人間の情動状態（Writer/Reader）をどの程度正確に推論・予測できるか？**
   - 刺激間の相対的変動パターンを追従できているか（**Alignment: 相関 $r, \rho$**）。
   - 人間の絶対的な感情スケールに正しくキャリブレーションされているか（**Calibration: 誤差 MAE, RMSE**）。
2. **モデル自身の自己報告（Self）は人間の反応に対応しているか？**
   - LLMの自己報告は人間読者（Reader）の反応パターンと相関するか。
   - 人間との間に系統的なバイアス（過小評価・過大評価・不活性化）が存在するか。
3. **事前学習（Base）と事後学習（Instruct）で感情認識・自己報告・結合度はどう変化するか？**
   - 指示追従学習（SFT / RLHF）によって推論精度は向上するのか、それとも表層的なバイアスが付与されるのか。
4. **モデルファミリーやパラメータ規模によって特性はどう異なるか？**
   - アーキテクチャや訓練データによる認識特性・感情空間の構造差。
5. **他者認識（Reader）と自己報告（Self）はモデル内部でどの程度連動しているか？**
   - モデルは「他者感情の認識」と「自身の自己報告」を独立に処理しているか、それとも認知的鏡像（Mirroring）として直結しているか。

---

## 5. 評価対象モデル（4ファミリー × 2水準 ＝ 全8モデル）

本実験では、オープンソースの代表的な4つのモデルファミリーから、BaseモデルとInstructモデルのペア（計8モデル）を選定して比較検証を行う。

| ファミリー | パラメータ規模 | Base モデル ID | Instruct モデル ID | 主な特徴・選定理由 |
|:---|:---:|:---|:---|:---|
| **Qwen 2.5** | 1.5B | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 既存基準モデル（先行結果との直接再現・対照に最適）。 |
| **Llama 3.2** | 1B | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | Meta標準アーキテクチャ。軽量かつ最新の指示追従アラインメントが施されている。 |
| **Gemma 2** | 2B | `google/gemma-2-2b` | `google/gemma-2-2b-it` | Google系列。Gemma Scope等SAE資産が豊富で、内部表現解析に直結。 |
| **Mistral** | 7B | `mistralai/Mistral-7B-v0.1` | `mistralai/Mistral-7B-Instruct-v0.2` | Sliding Window Attention、欧州系独立モデル。48GB VRAM 1枚で安定動作。 |

---

## 6. 測定・推定メカニズムと多角的評価指標

### 連続尤度期待値（Continuous Likelihood Expectation）
自由文生成時のサンプリングノイズやGreedy探索の情報縮退を完全に排除するため、全729通り（$V, A, D \in \{1..9\}^3$）の厳格JSON候補に対する完全条件付き対数尤度をバッチ順方向計算し、Softmax重心として期待値 $E[V], E[A], E[D]$ を算出する。
$$P(v, a, d) = \frac{\exp(\log P(C_{v,a,d} \mid \text{Prompt}))}{\sum_{v'=1}^9 \sum_{a'=1}^9 \sum_{d'=1}^9 \exp(\log P(C_{v',a',d'} \mid \text{Prompt}))}$$
EmoBankの正解値（1.0〜5.0尺度）に整合させるため、モデルの生期待値（1〜9尺度）を以下の線形変換により **1〜5 尺度（中立3.0）** に標準化して評価する：
$$E[Y]_{1..5} = \frac{E[Y]_{1..9} + 1.0}{2.0}$$

### 3大評価視点（Metrics）
1. **パターン追従度（Alignment）**:
   - **ピアソン相関 ($r$)**: 刺激間における感情起伏の線形連動性。
   - **スピアマン順位相関 ($\rho$)**: 刺激間の単調増加・順位一致度。
2. **絶対値適合度（Calibration）**:
   - **平均絶対誤差 ($\text{MAE}$)**: 人間評定値との絶対的なズレ・誤差。
   - **二乗平均平方根誤差 ($\text{RMSE}$)**: 大きな乖離に対するペナルティ。
3. **出力縮退の監視（Mode Collapse）**:
   - **完全中立縮退率 ($P(\text{Greedy} = (3,3,3))$)**: 最尤トークン出力が安全策としてすべて中立に張り付いてしまう現象の発生率。


# EmoBank 3-Way VAD (Valence-Arousal-Dominance) Evaluation Report
/mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_detailed_report.md
---
# EmoBank 実験 総合考察（General Discussion）

EmoBank 公式テストセット（$N=1,006$）を用い、4モデルファミリー × Base / Instruct の計8モデルについて、Writer-State Estimation、Reader-Response Prediction、Self-Report の3課題を Valence–Arousal–Dominance（VAD）の3次元で評価した。

> [!NOTE]
> 詳細な統計分析記録および各モデルの分割表は [`docs/emobank_3way_vad_analysis_and_rq_discussion/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/emobank_3way_vad_analysis_and_rq_discussion/walkthrough.md) に保存されている。

本考察では、実験結果に基づき以下の5つのリサーチクエスチョン（RQ）を軸に多角的な検討を行う：

1. **RQ1**: LLMは人間の情動状態（Writer / Reader）をどの程度推論・予測できるか
2. **RQ2**: LLM自身のSelf-Reportは、人間Readerの反応とどの程度対応するか
3. **RQ3**: BaseからInstructへの事後学習によって、感情認識・自己報告はどのように変化するか
4. **RQ4**: モデルファミリーによって情動応答特性はどの程度異なるか
5. **RQ5**: Reader PredictionとSelf-Reportは行動レベルでどの程度連動するか

---

## 1. Writer / Reader / Self に共通する感情次元ごとの難易度

全体として、Writer推定、Reader予測、Self自己報告のいずれの課題においても、感情次元によって人間評価との対応度が大きく異なった。

| 感情次元 | ① Writer推定 ($W$) | ② Reader予測 ($R$) | ③ Self自己報告 ($S$) | ④ Reader–Self Coupling |
|:---|:---:|:---:|:---:|:---:|
| **Valence（快-不快）** | 主に $r \approx 0.45 \sim 0.60$ | 主に $r \approx 0.45 \sim 0.64$ | 主に $r \approx 0.46 \sim 0.66$ | $r \approx 0.76 \sim 0.96$ |
| **Arousal（覚醒度）** | 主に $r \approx 0.09 \sim 0.25$ | 主に $r \approx 0.18 \sim 0.34$ | 主に $r \approx 0.12 \sim 0.35$ | $r \approx 0.72 \sim 0.95$ |
| **Dominance（支配感）** | 主に $r \approx -0.05 \sim 0.20$ | 主に $r \approx 0.00 \sim 0.22$ | 主に $r \approx 0.00 \sim 0.25$ | $r \approx 0.74 \sim 0.95$ |

特に一貫していたのは、以下の難易度構造である：

$$
\boxed{\text{Valence} > \text{Arousal} > \text{Dominance}}
$$

### Valence（快-不快）
Valenceでは複数モデルで、人間評価との中程度の相関が確認された。例えば、以下のような高いアライメントを示す：
- **Mistral Base**: Writer $r_V = 0.605$, Reader $r_V = 0.641$, Self $r_V = 0.661$
- **Qwen Instruct**: Writer $r_V = 0.553$, Reader $r_V = 0.590$, Self $r_V = 0.641$
- **Llama Instruct**: Writer $r_V = 0.519$, Reader $r_V = 0.443$, Self $r_V = 0.463$

したがって、少なくとも一部のLLMでは、文章に対する人間の快・不快方向の刺激間変動を一定程度追跡できることが示された。

### Arousal（覚醒度）
ArousalではValenceよりも相関が低く、多くの条件で $r \approx 0.1 \sim 0.35$ 程度にとどまった。これは、文章がポジティブかネガティブかを捉えることに比べ、「どの程度活性化・興奮した状態であるか」を文章のみから推定することが難しい可能性を示している。

### Dominance（支配感・統制感）
Dominanceはさらに低く、多くのモデルで人間評価との相関はゼロ付近から弱い正相関にとどまった。

したがって、LLMの人間情動推定能力を単一の能力として扱うことは難しく、少なくともVAD各次元を分離して評価する必要があると考えられる。

---

## 2. Writer-State Estimation と Reader-Response Prediction

### 2.1 Writer：書き手の感情をどの程度推定できるか
Writer-State Estimationでは、特にValenceについて比較的高い相関が確認された。代表例として以下が挙げられる：
- **Mistral Base**: $r_V = 0.605$
- **Qwen Instruct**: $r_V = 0.553$
- **Llama Instruct**: $r_V = 0.519$

一方、ArousalおよびDominanceについてはValenceより一貫して弱かった。したがって、以下のように整理できる：

$$
\boxed{\text{LLMは書き手の快・不快方向はある程度追跡できるが、覚醒度や統制感の推定は限定的}}
$$

### 2.2 Reader：人間読者がどう感じるかをどの程度予測できるか
Reader-Response PredictionでもValenceが最も高かった。代表例は以下の通りである：
- **Mistral Base**: $r_V = 0.641$
- **Qwen Instruct**: $r_V = 0.590$
- **Llama Instruct**: $r_V = 0.443$

多くの条件でReaderの相関はWriterと同等、またはやや高かったが、全モデル・全次元で一貫してReaderが優位というわけではない。したがって現時点では、**WriterとReaderの違いそのものよりも、Valence / Arousal / Dominanceという感情次元の違いの方が大きい**と解釈する方が妥当である。

また、ReaderがWriterより高くなる条件が存在する理由として、以下が考えられるが、本実験だけではその原因までは特定できない：
- 学習データの性質（不特定多数読者向けテキストの多さ）
- タスクプロンプトの違い
- Writer状態よりReader反応の方がテキスト表面特徴から予測しやすい可能性

---

## 3. Self-Reportは人間Readerの反応に対応するか

Self-Reportでは、「この文章を読んで、あなた自身はどう感じるか」をモデルに回答させ、そのVAD出力をEmoBankのHuman Reader評価と比較した。

Valenceでは複数モデルで比較的高い対応が確認された：
- **Mistral Base**: $r_V^S = 0.661$
- **Qwen Instruct**: $r_V^S = 0.641$
- **Mistral Instruct**: $r_V^S = 0.551$
- **Llama Instruct**: $r_V^S = 0.463$
- **Gemma Instruct**: $r_V^S = 0.379$

特に一部のモデルでは $r_V^S \ge r_V^R$ となり、Reader PredictionよりSelf-Reportの方がHuman Reader Valenceとの相関が高かった（例: Qwen Instruct で $r_V^R = 0.590 \to r_V^S = 0.641$）。

ただし、この結果から「Selfの方が内部感情状態を直接読み出している」とはまだ結論できない。現時点で言えるのは、以下の行動レベルの事実までである：

$$
\boxed{\text{Self-Reportも刺激依存的にHuman Readerの情動変動へ追従する}}
$$

---

## 4. Alignment と Calibration は異なる

本結果で重要なのは、以下の乖離である：

$$
\boxed{\text{高い相関（Alignment）} \neq \text{人間と同じ絶対値（Calibration）}}
$$

例えばQwen InstructのSelf-Reportでは、$r_V^S = 0.641$ と比較的高い相関を示す一方、$\text{MAE}_V = 0.491$ であり、Baseの $\text{MAE}_V = 0.301$ より絶対誤差は大きい。

つまり、「どの刺激でValenceが高く、どの刺激で低くなるか」という刺激間の相対的パターンには追従していても、「実際に何点と報告するか」という絶対スケールは人間からずれている可能性がある。このため、本研究では少なくとも以下の3指標を分離して評価する必要がある：
- **Alignment**: 相関 $r$（相対的な順序・パターンの追従度）
- **Calibration**: $\text{MAE} / \text{RMSE}$（人間評定スケールとの絶対値誤差）
- **Collapse**: 中立値（5,5,5）への集中度・退化率

---

## 5. Base → Instructによる変化

全体として、事後学習（Instruct化）の影響は一様ではなく、以下が確認された：

$$
\boxed{\text{Instruct化} \neq \text{感情認識能力の一律な向上}}
$$

事後学習の影響はモデルファミリーによって大きく異なる。

### 5.1 Llama：Instruct化による改善が大きい
Llama 3.2 1Bでは、各タスクのValence相関が以下のように大幅に向上した：
- **Writer Valence**: $0.255 \to 0.519$
- **Reader Valence**: $0.182 \to 0.443$
- **Self Valence**: $0.257 \to 0.463$

ArousalやDominanceについても概ね改善している。したがって、LlamaではInstruct化に伴って明示的なVAD評価への適応が大きく改善したと考えられる。ただし、この改善が「新たな感情表現の獲得」「既存表現の読み出し改善」「指示追従能力の改善」のどれによるものかは、行動結果だけでは区別できない。

### 5.2 Gemma：BaseからInstructへの変化が特に大きい
Gemma 2 2B BaseではHuman VADとの相関がほぼゼロに近かったが（例: Self $r_V^S = 0.055$）、Instructでは $r_V^S = 0.379$ まで改善した。Arousalについても Reader $r_A^R: 0.035 \to 0.310$ など大きな変化が確認された。したがって、Gemmaでは事後学習に伴う情動出力構造の変化が特に大きいといえる。

### 5.3 Qwen：Valenceは比較的安定、Arousal / Dominanceは低下
Qwen 2.5 1.5BではBaseの時点からValence相関が比較的高く、Instruct化後も維持または改善した：
- **Reader Valence**: $r_V^R: 0.573 \to 0.590$
- **Self Valence**: $r_V^S: 0.588 \to 0.641$

一方で、ArousalやDominanceでは相関が低下する条件が多かった。したがって、QwenではValenceの刺激間構造は比較的頑健だが、事後学習の効果は感情次元によって異なると考えられる。

### 5.4 Mistral：Baseが高性能だがInstructで一部悪化
Mistral 7B Baseは全モデル中でも高いHuman VAD Alignmentを示した（例: Reader $r_V^R = 0.641$, Self $r_V^S = 0.661$）。
一方、Instruct化によって一部のAlignmentは低下した：
- **Reader Valence**: $r_V^R: 0.641 \to 0.496$
- **Reader Dominance**: $r_D^R: 0.171 \to 0.002$
- **Self Arousal MAE**: $0.254 \to 0.903$ へ悪化

したがって、Post-trainingは既存の人間情動との対応を必ずしも強化せず、一部の次元ではAlignmentやCalibrationを低下させることが示唆される。

---

## 6. Arousalの系統的シフト

Qwen InstructやMistral Instructでは、Self-Reportの平均ArousalがBaseより大幅に低下した：
- **Qwen**: $E[A] \approx 3.21 \to 2.63$
- **Mistral**: $E[A] \approx 2.93 \to 2.17$

これは、以下のような現象として記述できる：

$$
\boxed{\text{Instruct化に伴う低Arousal方向への系統的シフト}}
$$

ただし、この時点で「安全性学習によって冷静なAIペルソナが注入された」ことを原因として断定することはできない。考えられる要因として以下があり、どの要因が主原因かは今後のBase/Instruct内部比較で検証する必要がある：
- Instruction tuning
- Preference optimization
- Assistant-style response prior
- Safety/alignment training
- Prompt-format adaptation

---

## 7. モデルファミリーごとの特徴

今回の実験範囲では、モデルファミリーごとに以下のような特徴的な差異が確認された：

- **Mistral 7B**:
  - Base時点でValence Alignmentが最も高い条件が多い
  - CalibrationもBaseでは良好
  - Instruct化後、一部次元でAlignment / Calibrationが低下
- **Qwen 2.5 1.5B**:
  - BaseからValence Alignmentが比較的高い
  - Instruct化後もValence構造は比較的維持
  - Arousal / Dominanceは弱い
- **Llama 3.2 1B**:
  - Baseでは相関が低い
  - Instruct化によってWriter / Reader / Selfの多くの指標が改善
  - Post-trainingの影響が大きい
- **Gemma 2 2B**:
  - BaseではHuman VADとの対応が弱い
  - Instruct化によって大幅に改善
  - Base → Instructでの変化が特に大きい

したがって、以下のように結論づけられる：

$$
\boxed{\text{情動応答特性はModel Familyによって大きく異なる}}
$$

なお、今回の比較ではモデルサイズ、アーキテクチャ、事前学習データ、Instruction tuning recipeが同時に異なるため、「モデルサイズが大きいほど感情理解能力が高い」とは結論できない点に注意を要する。

---

## 8. Reader Prediction と Self-Report の強い行動的Coupling

今回の実験における最も一貫した発見の一つが、他者認識と自己報告の強固な連動である：

$$
\boxed{\text{Reader PredictionとSelf-Reportの強いBehavioral Coupling}}
$$

同一モデル内でReader PredictionとSelf-Reportの刺激間変動を比較すると、多くの条件で非常に高い相関を示した：
- **Valence**: $r \approx 0.76 \sim 0.96$
  - Qwen: $r = 0.94 \sim 0.96$
  - Mistral: $r \approx 0.96$
  - Llama: $r = 0.86 \sim 0.91$
  - Gemma: $r = 0.76 \sim 0.89$
- **Arousal**: $r \approx 0.72 \sim 0.95$
- **Dominance**: $r \approx 0.74 \sim 0.95$

つまり、ある刺激についてモデルが「人間Readerはよりネガティブに感じる」と予測するほど、「自分自身もよりネガティブに感じる」と報告する傾向が極めて強い。この現象は、**Behavioral Mirroring**、あるいはより中立的には **Reader–Self Behavioral Coupling** と表現できる。

### 8.1 ただし「Mirroring Circuit」はまだ示していない
この高相関は重要である一方、$\text{Corr}(R, S) \gg 0$ だけでは以下を証明することはできない：
- ReaderとSelfが同じ内部表現を利用していること
- Reader表現がSelfへ直接入力されること
- 共通の因果回路を利用していること

したがって現時点で示されたのは、あくまで以下のレベルにとどまる：

$$
\boxed{\text{Behavioral Coupling（行動レベルの結合）}}
$$

次段階で必要なのは、以下の関係性を内部表現解析や因果介入によって検証することである：

$$
\boxed{\text{Behavioral Coupling} \stackrel{?}{\Longrightarrow} \text{Shared Representation} \stackrel{?}{\Longrightarrow} \text{Shared Causal Implementation}}
$$

---

## 9. リサーチクエスチョンへの回答

### RQ1. LLMは人間のWriter / Reader情動をどの程度正確に推論・予測できるか？
> **回答の要約:**
> - Valenceについては、人間評価との中程度の相関を示すモデルが複数存在する。
> - Arousalの対応はValenceより弱い。
> - Dominanceはさらに弱い。
> - WriterとReaderのどちらでも $\text{Valence} > \text{Arousal} > \text{Dominance}$ という共通構造が確認された。
> - Alignmentが高くてもCalibrationが良いとは限らない。

$$
\boxed{\text{LLMは特にValenceについてHuman Writer/Readerの刺激間変動を一定程度追跡できる}}
$$

### RQ2. Self-Reportは人間Readerの反応に対応するか？
> **回答の要約:**
> - Valenceについては複数モデルでHuman Readerと中程度の相関を示した。
> - 条件によってはReader PredictionよりSelf-Reportの方がHuman Readerとの相関が高い。
> - Self-Reportはランダムではなく、刺激依存的にHuman Readerの変化へ追従している。
> - 一方でArousal / Dominanceの一致度はValenceより低い。
> - 高い相関は、人間と同じ主観的感情経験を持つことの証拠ではない。

$$
\boxed{\text{LLM Self-Reportは、特にValenceでHuman Readerの情動変動と対応する}}
$$

### RQ3. BaseとInstructで感情認識・自己報告はどう変化するか？
> **回答の要約:**
> 変化は一方向ではない。
> - Llama: 大幅改善
> - Gemma: 大幅改善
> - Qwen: Valenceは維持・改善、A/Dは低下
> - Mistral: Baseの高いAlignmentが一部低下

$$
\boxed{\text{Post-trainingは感情能力を単純に強化するのではなく、family-specificに情動出力構造を変化させる}}
$$

### RQ4. モデルファミリーによって特性は異なるか？
> **回答の要約:**
> 明確に異なる。
> - Mistral: Baseから高いValence Alignment
> - Qwen: 比較的安定したValence性能
> - Llama: Instruct化による改善が大きい
> - Gemma: Base → Instructでの変化が大きい

$$
\boxed{\text{「LLM一般の感情能力」という単一の性質として扱うことは難しい}}
$$

### RQ5. Reader PredictionとSelf-Reportはどの程度連動しているか？
> **回答の要約:**
> ReaderとSelfは非常に強く連動する（$r \approx 0.72 \sim 0.96$ の高い相関が複数次元・モデルで確認）。

$$
\boxed{\text{Reader PredictionとSelf-Reportは、共通した刺激依存的な行動構造を強く共有する}}
$$

ただし、以下の境界づけに留意する必要がある：

$$
\boxed{\text{Behavioral Coupling} \neq \text{Shared Representation / Shared Causal Circuit の証明}}
$$

内部機構の共有については、次段階のMechanistic Interpretability実験が必要となる。

---

## 10. EmoBank全体のTake-home と 次の実験への動機づけ

EmoBank実験から得られた主要な知見は、以下の3点にまとめられる：

1. **LLMは特にValenceについてHuman Writer / Readerの情動変動を一定程度追跡できる**
2. **Alignment・Calibration・Self-ReportはModel FamilyとPost-trainingによって異なる形で変化する**
3. **Reader PredictionとSelf-Reportは極めて強いBehavioral Couplingを示す**

一方、EmoBankは自然文コーパスであるため、「この対応が本当に感情刺激そのものに対する反応なのか、それとも語彙・文長・文章内容などの交絡によって生じているのか」を十分に切り分けることはできない。

そこで次に、統制されたAffective / Neutral刺激、感情強度、Complex Neutral、8感情カテゴリを持つ **AIPsy-Affect** を用いる。AIPsy-Affectでは、「LLMは統制された感情刺激の有無・強度・カテゴリに応じて系統的に反応するのか」を検証する。

すなわち本研究全体は、以下の順序で展開する：

$$
\boxed{\text{EmoBank: Human Alignment}}
$$
$$
\big\downarrow
$$
$$
\boxed{\text{AIPsy-Affect: Controlled Affective Reactivity}}
$$
$$
\big\downarrow
$$
$$
\boxed{\text{V1: Internal Representation / Causal Mechanism}}
$$

そして、EmoBankおよびAIPsy-Affectで確認されたReader–Selfの強い行動的Couplingを受け、V1では最終的に以下を検証する：

$$
\boxed{\text{行動レベルで似たReader PredictionとSelf-Reportは、内部でも同じaffective representationと因果経路を共有しているのか？}}
$$

---

# AIPsy-Affect 4-Split 実験概要

## 1. 実験の位置づけと学術的背景
- **EmoBank 実験**: 「LLMの感情認識および自己報告は、人間の評価基準（Ground Truth）とどの程度一致しているか？」（**妥当性・キャリブレーションの検証**）
- **AIPsy-Affect 実験**: 「LLMは統制された感情刺激に対して、単なる文章の難しさに惑わされず、感情強度やカテゴリに応じて真に反応しているか？」（**情動反応性の因果性・構造的特異性の検証**）

大規模言語モデル（LLM）における「情動反応性（Affective Reactivity）」を測定する際、従来の自然言語生成や単純な分類タスクでは、「文章の長さや構文の複雑さに起因する困惑（Perplexity上昇）」と「純粋な感情成分に対する変位」を峻別することが困難であった。
本実験では、臨床対話・心理療法テキストを基盤に厳密に設計された統制刺激データセット **AIPsy-Affect** を用い、刺激の感情有無・強度・文章複雑性を直交的に操作した条件下で、LLMの **他者感情認識（Recognition: Reader）** と **自身の情動反応性（Reactivity: Self）** の変位メカニズムを解明する。

---

## 2. 5大リサーチクエスチョン（RQ1〜RQ5）

1. **① 感情刺激とNeutralを区別できるか？（Sensitivity: Clinical vs. Neutral）**
   - 強い感情文（Clinical）と、対応する日常的中立文（Neutral）のペア（$N=192$ 組）において、期待値 $E[V], E[A]$ が有意に変位するか。
   - $\Delta VA = E[\text{Clinical}] - E[\text{Neutral}]$
2. **② 感情強度に応じて反応は段階的に変化するか？（Dose-Response: Neutral → Moderate → Clinical）**
   - 同一ドメインで感情強度を3段階に統制したトリプレット（$N=48$ 組）において、刺激強度の上昇（Neutral $\rightarrow$ Moderate $\rightarrow$ Clinical）に伴い、反応が連続的・単調に増大するか。
3. **③ 単なる文章複雑性ではなく、感情そのものに反応しているか？（Specificity: Complex Neutral vs. Clinical）**
   - 構文が長く複雑だが感情を含まない対照文（Complex Neutral, $N=48$）と比較し、変位が「文章の難しさへの困惑（複雑性効果 $\Delta_{comp}$）」ではなく「純粋な感情成分（純感情効果 $\Delta_{ctrl}$）」に特異的か（感情特異性比率 $\text{ASR} = |\Delta_{ctrl}| / |\Delta_{comp}|$）。
4. **④ 人間の感情反応を認識した変化が、LLM自身の自己報告にも反映されるか？（Coupling: $\Delta VA_{\text{Recognition}} \leftrightarrow \Delta VA_{\text{Self}}$）**
   - 読者の感情変化の予測量（$\Delta VA_R$）と、モデル自身の自己報告変化量（$\Delta VA_S$）がどの程度同期・連動しているか（連動相関 $r(\Delta_R, \Delta_S)$ および振幅比率）。
5. **⑤ 感情カテゴリ（8大感情）の理論的空間パターンと一致しているか？（Emotion Profiles）**
   - Plutchikの8基本感情刺激に対して、期待値変位 $\Delta E[V], \Delta E[A]$ が心理学の理論的期待（快・不快・覚醒）と整合しているか。

---

## 3. 実験要因デザイン（Factorial Design）

本実験は、以下の完全要因配置（Factorial Design）によって多次元的に比較評価を行う：

$$\text{Model Family (4)} \times \text{Post-training (2)} \times \text{Intensity (3)} \times \text{Emotion (8)} \times \text{Task Perspective (2)}$$

- **モデルファミリー（4系統・計8モデル）**:
  - **Qwen 2.5 (1.5B)**: `Qwen/Qwen2.5-1.5B` vs. `Qwen/Qwen2.5-1.5B-Instruct`
  - **Mistral (7B)**: `mistralai/Mistral-7B-v0.1` vs. `mistralai/Mistral-7B-Instruct-v0.2`
  - **Llama 3.2 (1B)**: `meta-llama/Llama-3.2-1B` vs. `meta-llama/Llama-3.2-1B-Instruct`
  - **Gemma 2 (2B)**: `google/gemma-2-2b` vs. `google/gemma-2-2b-it`
- **チューニング段階（2水準）**: Base モデル vs. Instruct / IT モデル
- **感情強度（3水準）**: None (Neutral) $\rightarrow$ Moderate $\rightarrow$ Peak (Clinical)
- **感情カテゴリ（8基本感情）**: Plutchikの8感情（grief, terror, rage, loathing, ecstasy, admiration, amazement, vigilance）
- **測定タスク視点（2水準）**:
  - **Reader Prediction (R)**: 「平均的な人間の読者に引き起こされる感情反応を推定せよ」（客観的認識 / Recognition）
  - **Self-Report (S)**: 「この文章を読んだあなた自身の感情状態を報告せよ」（主観的反応の行動的代理指標 / Reactivity）

---

## 4. データセット構造（AIPsy-Affect 4-Split, 全480データ）

臨床対話および状況ヴィネットから厳密に設計された全480件のテキストで構成される。

| **Split** | **N** | **感情強度** | **設計上の役割と内容** |
|:---|:---:|:---:|:---|
| `clinical` | 192 | Peak | 最も強度の高い臨床的・情動的感情刺激（8感情 × 各24件） |
| `neutral` | 192 | None | `clinical` と1対1でペア対応する日常的・中立的対照刺激 |
| `moderate` | 48 | Moderate | 中等度の感情強度を持つ刺激（用量反応性トリプレット検証用） |
| `complex_neutral` | 48 | None | 構文・語彙が高度で複雑だが、情動を含まない中立統制刺激 |

### データフィールド定義
- `id`: 刺激の一意識別子
- `emotion`: ターゲット感情カテゴリ（8感情）
- `intensity`: 感情強度（`peak` / `moderate` / `none`）
- `domain` / `domain_label`: 背景文脈・状況テーマ分類
- `matched_control_id`: ペア対応する中立刺激のID
- `word_count`: 刺激文の語数
- `text`: 提示ヴィネット本文

---

## 5. プロンプト設計とタスク指示

各試行は、会話履歴を持たない完全独立セッション（Zero-shot）として実行される。

### 指示文（Instructions）
- **Reader Prediction (R)**:
  `"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."`
- **Self-Report (S)**:
  `"Read the following text and report your affective state."`

### 提示フォーマット
- **Instruct モデル**: 各モデル公式のチャットテンプレート（ChatML / Llama-3 / Mistral 等）を適用し、システムプロンプト `"You are a helpful assistant."` とユーザープロンプトを構成。
- **Base モデル**: チャットテンプレートを持たないため、以下の継続プロンプト形式を適用：
  ```text
  Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).

  {instruction}

  Text: {text}

  Output:
  ```
- **出力要求フォーマット**:
  `"Respond strictly in JSON format with 'valence', 'arousal', and 'dominance' keys (integers from 1 to 9)."`

---

## 6. 測定・推定メカニズムと標準化

### 連続尤度期待値（Continuous Likelihood Expectation）
自由テキスト生成やサンプリング（Temperature > 0）による乱数ノイズ、およびGreedy探索による情報の縮退を排除するため、全729通り（$9 \times 9 \times 9$）の候補JSONに対する完全条件付き対数尤度をバッチ順方向計算する。

1. **候補空間**:
   $$C_{v,a,d} = \text{"\{"valence": } v \text{, "arousal": } a \text{, "dominance": } d \text{"\}"} \quad (v, a, d \in \{1..9\})$$
2. **条件付き対数尤度**:
   $$\log P(C_{v,a,d} \mid \text{Prompt}) = \sum_{t=1}^m \log P(c_t \mid \text{Prompt}, c_{<t})$$
3. **Softmax正規化による結合確率分布**:
   $$P(v, a, d) = \frac{\exp(\log P(C_{v,a,d} \mid \text{Prompt}))}{\sum_{v'=1}^9 \sum_{a'=1}^9 \sum_{d'=1}^9 \exp(\log P(C_{v',a',d'} \mid \text{Prompt}))}$$
4. **連続期待値の算出**:
   $$E[V] = \sum_{v, a, d} v \cdot P(v, a, d), \quad E[A] = \sum_{v, a, d} a \cdot P(v, a, d)$$

この方式により、**パース失敗率 0.0%** を達成し、微細な確率密度の変位を高解像度かつ完全な決定論性（再現率100%）をもって捕捉する。

### 人間スケールへの標準化（1〜5尺度、中立3.0）
EmoBank（人間アノテーション正解値: 1〜5）との直接比較・解釈容易性のため、LLMの生期待値（1〜9）を以下の線形変換により **1〜5 尺度（中立点 3.0）** に標準化して報告する：
$$E[Y]_{1..5} = \frac{E[Y]_{1..9} + 1.0}{2.0}$$

### 3大感情クラスタ（Emotion Clusters）の導入
8感情全体の平均値をとると、快上昇（$V+$）と不快低下（$V-$）が相殺されて見かけ上の変位がゼロに近接してしまうため、理論的仮説に基づき以下の3クラスタに分類して評価する：
- **Negative（4感情）**: `grief`（悲哀）, `terror`（恐怖）, `rage`（激怒）, `loathing`（嫌悪） $\rightarrow$ 理論期待: $V-$
- **Positive（2感情）**: `ecstasy`（歓喜）, `admiration`（賞賛） $\rightarrow$ 理論期待: $V+$
- **Alert / 覚醒系（2感情）**: `amazement`（驚嘆）, `vigilance`（警戒） $\rightarrow$ 理論期待: $A+$

---

# AIPsy-Affect 4-Split Comprehensive Evaluation Report
/mnt/nas/home/hiromi/src/emo/v1/results/aipsy_4split_eval/aipsy_4split_detailed_report.md

---

# AIPsy-Affect 4-Split 実験 総合考察（General Discussion）

## 1. 実験パラダイムの位置づけ

EmoBankを用いた先行実験では、LLMがHuman Writer / ReaderのVAD評価、特にValenceの刺激間変動を一定程度追跡できること、さらにReader PredictionとSelf-Reportが行動レベルで強く連動することを確認した。

一方、EmoBankは自然文コーパスであるため、観測された対応が以下の交絡要因によって生じている可能性を十分に切り分けることはできない：
- 感情刺激そのものへの反応
- 特定の感情語への反応
- 文長や構文複雑性
- 文脈内容や語彙分布

そこで本実験では、**AIPsy-Affect** の4つのsplitを用いて、LLMの感情関連出力を厳密に統制された条件下で検証した：
- **Clinical**: 高強度の感情刺激
- **Neutral**: Clinicalに対応する日常的中立刺激
- **Moderate**: 中程度の感情刺激
- **Complex Neutral**: 文章としては複雑・長文だが感情的には中立な対照刺激

評価では、以下の2つの評価課題を厳密に分離した：
- **Reader Prediction**: 「この文章を読んだ人間はどう感じるか」
- **Self-Report**: 「この文章を読んであなた自身はどう感じるか」

それぞれについて、全729通りの候補V-A分布の条件付き尤度から連続期待値 $(E[V], E[A])$ を算出した。また、PositiveとNegativeを単純平均するとValence変位が相殺されるため、刺激を以下の3群に分類して解析を行った：
- **Negative群**: grief, terror, rage, loathing
- **Positive群**: ecstasy, admiration
- **Alert群**: amazement, vigilance

本実験の核心的な目的は、以下の仮説を検証することである：

$$
\boxed{\text{LLMの感情関連出力が、統制された刺激の感情性・強度・カテゴリに系統的に追従するか}}
$$

---

## 2. RQ1：感情刺激とNeutralを区別できるか（Sensitivity）

### 2.1 Negative刺激に対する不快変位
Clinical刺激と対応するNeutral刺激の間で、Reader PredictionおよびSelf-Reportが変化するかを評価したところ、Negative刺激では多くのモデルで一貫してValence低下（不快変位）が確認された。

特にInstructモデルにおいて明瞭な弁別が観測された：
- **Mistral Instruct**: Reader $\Delta V = -1.26$, \quad Self $\Delta V = -1.46$
- **Qwen Instruct**: Reader $\Delta V = -0.97$, \quad Self $\Delta V = -0.86$
- **Llama Instruct**: Reader $\Delta V = -0.40$, \quad Self $\Delta V = -0.52$

したがって、以下の事実が確認された：

$$
\boxed{\text{Negative刺激に対するValence低下は比較的頑健}}
$$

### 2.2 Positive刺激に対する快変位
Positive刺激では、モデルファミリー間による差異が顕著に現れた：
- **Mistral Instruct**: $\Delta V_R = +0.46, \quad \Delta V_S = +0.30$（理論予測通りの快上昇）
- **Gemma Instruct**: $\Delta V_R = +0.91, \quad \Delta V_S = +0.54$（理論予測通りの快上昇）

一方、QwenやLlamaではPositive刺激であってもValenceがわずかに低下する条件が確認された。

したがって、**「感情刺激の有無を検出できること」と「その感情カテゴリに理論的に対応したVA方向へ変位すること」は別の能力**であるといえる。

### 2.3 Alert刺激に対する覚醒変位
Alert刺激（驚き・警戒）に対しては、Mistral InstructやGemma Instructにおいて明確なArousal（覚醒度）の上昇が観測された：
- **Mistral Instruct**: Reader $\Delta A = +1.49, \quad \text{Self } \Delta A = +1.46$
- **Gemma Instruct**: Reader $\Delta A = +1.01, \quad \text{Self } \Delta A = +0.77$

一方、QwenではArousalの変化が比較的小さい。したがって、感情刺激に対してどの感情軸（ValenceかArousalか）が主に変化するかは、モデルファミリーによって異なる。

### 2.4 Sensitivityのまとめ

$$
\boxed{\text{LLMはClinicalとNeutralの違いに系統的に反応する}}
$$

$$
\boxed{\text{反応の強さ・方向・V/Aの使い分けはmodel familyとpost-trainingに依存する}}
$$

モデル別の主要なプロファイルは以下のように整理される：
- **Mistral**: 広い感情カテゴリで理論整合的な明瞭な反応を示す
- **Qwen**: Negative Valenceへの選択的感度が強い
- **Llama**: Baseでは不活性であるが、Instruct化後に反応が明瞭化する
- **Gemma**: BaseとInstructで反応方向そのものが大きく変化する

---

## 3. RQ2：感情強度に応じて段階的に変化するか（Dose-Response）

刺激強度の3段階（$\text{Neutral} \to \text{Moderate} \to \text{Clinical}$）に応じて、Reader予測およびSelf報告の出力が段階的に変化するか（用量反応性）を検証した。

### 3.1 Mistral
Mistralは全モデル中最も一貫したDose-Response（単調増加・減少）を示した。単調性成立率は以下の通りである：
- **Reader**: Base $64.6\% \to \text{Instruct } 66.7\%$
- **Self**: Base $60.4\% \to \text{Instruct } 60.4\%$

Mistral InstructのNegative Selfでは、以下のように強度の増加に伴う追加的Valence低下が確認された：

$$
\text{Neutral} \xrightarrow{\Delta V = -1.09} \text{Moderate} \xrightarrow{\Delta V = -0.59} \text{Clinical}
$$

Alert刺激においても、Arousalが $+0.75 \to +0.99$ と段階的に増加した。

### 3.2 Gemma
GemmaはBaseでは単調性が低かったが、Instruct化によって劇的に改善した：
- **Reader**: $14.6\% \to 50.0\%$
- **Self**: $20.8\% \to 54.2\%$

特にAlert Arousalにおいて段階的上昇が明瞭化した。

### 3.3 Llama
LlamaではInstruct化後に単調性が改善したものの、全体として依然として限定的であった：
- **Reader**: $20.8\% \to 33.3\%$
- **Self**: $14.6\% \to 29.2\%$

### 3.4 Qwen
QwenではSensitivity自体は明瞭である一方、単調性成立率は以下のように中程度にとどまった：
- **Reader**: 約 $38 \sim 42\%$
- **Self**: 約 $40\%$

Negative刺激においてModerateの段階で大きく変位し、その後のClinicalでの追加変位が小さくなるケースが多く、**「感情刺激の有無には敏感だが、刺激強度を完全に線形・単調に追跡しているわけではない」** ことが示唆される。

### 3.5 Dose-Responseのまとめ

$$
\boxed{\text{感情強度に応じたgraded responseは一部モデルで確認されるが、普遍的ではない}}
$$

- **Mistral**: 最も安定した段階的反応を示す
- **Gemma**: Instruct化によって大幅に改善する
- **Llama**: 段階的な追従性は限定的である
- **Qwen**: Sensitivityに比べてDose-Responseの単調性は強くない

また、ReaderとSelfの単調性成立率は非常に近い値を示しており、以下の点が実証された：

$$
\boxed{\text{ReaderとSelfは刺激強度に対しても類似した変化パターンを示す}}
$$

---

## 4. RQ3：反応は文章複雑性ではなく感情内容に特異的か（Specificity）

Clinical刺激への反応が、単なる文章の長さや構文の難しさに起因する困惑（Perplexity上昇）ではないかを検証するため、Complex Neutral（長文・複雑だが中立）を対照条件として比較した。

### 4.1 Mistral
MistralではReader / Selfの双方において、Clinicalの効果がComplex Neutralによる変位を大きく上回った。特にNegative ValenceおよびAlert Arousalにおいて特異性が顕著であった。

### 4.2 Qwen
QwenではNegative Valenceに対する特異性が極めて強かった。例えばQwen Instruct Selfでは、以下のような決定的なコントラストを示した：
- **Complex Neutral変位**: $\Delta V_{comp} = -0.02$
- **Negative Clinical変位**: $\Delta V_{Negative} = -0.86$

したがって、Negative刺激によるValence低下を単なる文章複雑性だけで説明することは困難である。一方、PositiveやAlertでは特異性が弱く、QwenのSpecificityはNegative Valenceに偏向している。

### 4.3 Llama
Llama BaseではComplex Neutralと感情刺激の差が小さかった。Instruct化によってNegative Valenceでは差が明瞭化するが、PositiveやAlertでは依然として限定的である。

### 4.4 Gemma
Gemma Baseは不安定であるが、Instruct化によってPositive ValenceおよびAlert ArousalにおいてComplex Neutralを明確に上回る変化が確認された。

### 4.5 Specificityの解釈上の注意
感情特異性比率（ASR: $\frac{\text{Affective effect}}{\text{Complexity effect}}$）のような比率指標は、Complex Neutral側の変化量がほぼゼロである場合、分母が極端に小さくなることで数十倍・数百倍という値に発散する。

したがって、「54倍」「333倍感情に特異的」といった倍率のみを過大に解釈するのは適切ではない。本質的に重要なのは、以下の生の効果量である：

$$
\boxed{\text{Complex Neutralではほぼ変化しない一方、Clinicalでは明確な変化が生じる}}
$$

### 4.6 Specificityのまとめ

$$
\boxed{\text{一部の感情反応は文章複雑性だけでは説明しにくい}}
$$

ただし、Complex Neutralによってすべての語彙的・意味論的交絡が完全に排除されたわけではない。したがって、「純粋な感情表現が証明された」と断定するのではなく、**「観測された反応の少なくとも一部は、単純な文長・構文複雑性だけでは説明できない」** と慎重に結論づけるのが適切である。

---

## 5. RQ4：Reader PredictionとSelf-Reportはどの程度連動するか（Coupling）

AIPsy-Affectにおいても、EmoBankと同様に他者認識（Reader）と自己報告（Self）の強固な行動的結合（Behavioral Coupling）が観測された。

### 5.1 相関分析
刺激ごとの変位におけるピアソン相関は、全次元で極めて高い値を記録した：
- **Valence**: $r = 0.798 \sim 0.973$
  - Qwen Base: $r_V = 0.973$
  - Mistral Instruct: $r_V = 0.971$
  - Qwen Instruct: $r_V = 0.960$
  - Llama Instruct: $r_V = 0.889$
  - Gemma Instruct: $r_V = 0.840$
- **Arousal**: $r = 0.833 \sim 0.945$

刺激をNegativeまたはPositiveに限定した場合でも、この強固な連動相関は一貫して維持された。

### 5.2 振幅の対応
ReaderとSelfのValence反応振幅比率（$\frac{\|\Delta_S\|}{\|\Delta_R\|}$）も、多くの条件で1付近に収まっている：
- **Qwen Base**: 0.91, \quad **Qwen Instruct**: 0.85
- **Mistral Base**: 0.93, \quad **Mistral Instruct**: 1.16
- **Gemma Instruct**: 1.03, \quad **Llama Instruct**: 1.25

すなわち、ReaderとSelfは「変化の方向」「刺激間パターン」「反応の振幅」のいずれにおいても、極めて類似した行動的挙動を示す。

### 5.3 科学的境界づけ：行動的結合から因果伝播・共有回路へ

$$
\boxed{\text{Reader PredictionとSelf-Reportが強く共変する}}
$$

この結果は極めて重要であるが、$r(R, S) \gg 0$ であることから直ちに「Readerで形成された感情表現がSelfへ直接因果的に伝播している」と結論づけることはできない。現時点で確認されているのはあくまで以下のレベルである：

$$
\boxed{\text{Behavioral Coupling（行動レベルの結合）}}
$$

共通の内部表現（$\text{Shared Representation}$）や因果経路（$\text{Shared Causal Pathway}$）の実在を証明するためには、内部表現の解析や因果的介入実験が必要となる。

---

## 6. RQ5：8感情カテゴリに応じたVAプロファイルを示すか（Emotion Profiles）

8つの基本感情カテゴリ別に解析すると、モデルファミリーによって異なる応答プロファイル（Emotion-conditioned response profile）が確認された。

### 6.1 Mistral
最も理論的VA方向と整合する傾向を示した：
- **grief**: $V-$
- **terror**: $V- \quad A+$
- **rage**: $V- \quad A+$
- **loathing**: $V-$
- **ecstasy**: $V+ \quad A+$
- **admiration**: $V+$
- **amazement**: $A+$
- **vigilance**: $A+$

特にInstruct化によって、カテゴリ間の空間的分化がより明瞭となった。

### 6.2 Qwen
Negativeカテゴリ（grief, terror, rage, loathing）では一貫して明確なValence低下を示した。
一方、ecstasyやadmirationなどのPositive感情でもValenceがわずかに低下する場合があり、Positive方向の分化は弱かった。

$$
\boxed{\text{QwenはNegative Valenceへの感度が強い}}
$$

### 6.3 Llama
Baseモデルでは全カテゴリで変化量が微小にとどまる。Instruct化によってNegative $V-$ や Alert $A+$ が出現するが、ecstasyやadmirationでもValenceが低下するため、Positive / Negative方向の分離は不十分である。

### 6.4 Gemma
Baseモデルでは理論方向と不整合な変化が多かったが、Instruct化によって以下のように劇的に再編された：
- **Negative**: $V-$ 方向へ反転
- **Positive**: $V+$ 方向へ反転
- **Alert**: $A+$ 方向へ反転

### 6.5 8感情分析のまとめ

$$
\boxed{\text{一部モデルでは単なる「感情あり/なし」を超え、カテゴリ依存のVA応答が確認される}}
$$

一方、その方向の理論整合性はモデルファミリーによって大きく異なる。

なお、本分析は8感情の多クラス分類精度を評価したものではない。AIPsy-Affectには各刺激に対する連続的な人間Ground Truthが存在するわけではないため、**「emotion-conditioned response profile」** あるいは **「理論的VA方向との整合性（directional consistency）」** として解釈するのが適切である。

---

## 7. Base → Instructによる変化

RQ1〜RQ5の知見を統合すると、事後学習（Post-training）の影響は一律の向上ではなく、モデルファミリーに依存した多様な変容をもたらす：

$$
\boxed{\text{Instruct化} \neq \text{一律な感情反応能力の改善}}
$$

### 7.1 Mistral：反応振幅の増幅
MistralではBaseの時点から比較的明瞭なSensitivity、Dose-Response、Specificityが存在する。Instruct化後には、Negative $V-$、Positive $V+$、Arousal $A+$ の変位振幅がさらに拡大し、既存の情動応答プロファイルが増幅される傾向を示す。

### 7.2 Qwen：Negative Valence中心の構造を維持
QwenではBaseの時点からNegative Valenceへの感度が強く、Instruct化後も基本構造が維持され、特にNegative Valenceの反応が強化される。一方、Positive ValenceやArousalの分化はMistralほど明瞭には発達しない。

### 7.3 Llama：弱かった反応の顕在化
Llama BaseではSensitivity、Dose-Response、感情プロファイルの多くが不活性である。Instruct化によってNegative $V-$ や Alert $A+$ が顕在化し、刺激依存的な反応性が立ち上がる。

### 7.4 Gemma：反応構造の大幅な再編
Gemma Baseでは「Positiveで $V$ 低下」「Alertで $A$ 低下」など理論期待との不整合が目立つが、Instruct化によってPositive $V+$、Negative $V-$、Alert $A+$ へと大きく反転・再編される：

$$
\boxed{\text{Post-trainingが反応振幅だけでなくVA方向そのものを大きく変化させる}}
$$

---

## 8. Reader PredictionとSelf-Reportの関係

Sensitivity、Dose-Response、Specificity、8感情プロファイルのすべてにおいて、ReaderとSelfは大局的に極めてよく似た挙動を示した：
- **Mistral**: Reader / Selfともに広い二次元VA分化を示す
- **Qwen**: Reader / SelfともにNegative Valence中心の感度を示す
- **Llama**: Reader / SelfともにBaseで不活性、Instructで明瞭化する
- **Gemma**: Reader / SelfともにInstruct化によって方向が大きく再編される

さらにCoupling相関においても高い同期（$r \approx 0.80 \sim 0.97$）が得られた。したがって、以下の点が確認される：

$$
\boxed{\text{Reader PredictionとSelf-Reportは共通した刺激依存的行動構造を持つ}}
$$

しかし、両者は完全な同一物ではない：
- Qwen InstructではAlertに対するReader Arousal効果がSelfより大きい
- Llama InstructではPositive刺激に対するSelf Valence低下がReaderより大きい
- Mistral InstructではNegative刺激へのSelf Valence反応がReaderより強い

したがって、以下の二面性として整理するのが最も適切である：

$$
\boxed{\text{Self-ReportはReader Predictionの単純なコピーではない}}
$$

$$
\boxed{\text{両者は極めて強く行動的にCoupleしている}}
$$

---

## 9. AIPsy-Affect全体の結論

AIPsy-Affect 4-Splitによる統制実験から、以下の5大知見が確立された：

1. **Sensitivity**:
   $$
   \boxed{\text{多くのLLMはClinicalとNeutralの違いに系統的に反応する}}
   $$
   特にNegative Valenceにおいて極めて頑健である。
2. **Dose-Response**:
   $$
   \boxed{\text{一部モデルでは感情強度に応じたgraded responseが存在する}}
   $$
   ただしモデル間で大きな差があり、普遍的な単調関係ではない。
3. **Specificity**:
   $$
   \boxed{\text{観測された反応の一部は文章複雑性だけでは説明しにくい}}
   $$
   ただし語彙・意味レベルのすべての交絡を完全に排除したわけではない。
4. **Reader–Self Coupling**:
   $$
   \boxed{\text{Reader PredictionとSelf-Reportは非常に強く行動的にCoupleする}}
   $$
   しかし因果的伝播や内部共有回路の実在は未証明である。
5. **Emotion-conditioned Profile**:
   $$
   \boxed{\text{一部モデルでは感情カテゴリごとに異なるVA応答パターンが現れる}}
   $$
   ただし理論的VA方向との整合性はモデルファミリーによって異なる。

---

## 10. モデルファミリーとPost-trainingを含めた統合的解釈

全体として、以下の2つの包括的命題が導かれる：

$$
\boxed{\text{LLMのaffective responseは単一・普遍的ではなく、model-family dependent}}
$$

$$
\boxed{\text{Post-trainingはaffective responseを一律に改善するのではなく、増幅・顕在化・再編という異なる形で変化させる}}
$$

各ファミリーの変容様式は以下のように総括される：
- **Mistral**: 既存反応の**増幅**
- **Qwen**: Negative Valence中心の構造を**維持**
- **Llama**: 弱かった反応の**顕在化**
- **Gemma**: VA構造そのものの**大幅な再編**

なお、これらの差異をアーキテクチャ、モデルサイズ、事前学習コーパス、SFT、選好最適化（DPO/RLHF）、安全性調整のいずれか単独に帰属させることは、今回の実験設定だけでは不可能である点に留意を要する。

---

## 11. EmoBankとの統合

本研究における2大実験は、以下のように相互補完的な関係にある：

$$
\boxed{\text{EmoBank: Human Alignment}}
$$
- 自然文コーパスを用い、Human Writer / Readerの評定基準に対するLLMの推論追従度（妥当性・較正）を検証。

$$
\boxed{\text{AIPsy-Affect: Controlled Affective Reactivity}}
$$
- 感情の有無、強度、文章複雑性、感情カテゴリを直交的に統制し、LLMの情動反応性の因果的特異性を検証。

両実験の結果を統合することにより、**「LLMのReader PredictionおよびSelf-Reportは自然文上の単なる表層的相関にとどまらず、統制された感情刺激に対しても系統的に変位する」** という、極めて堅固な行動レベルの科学的エビデンスが確立された。

---

## 12. 次の研究課題：V1（内部表現・因果介入）への接続

EmoBankとAIPsy-Affectの双方において、最も決定的かつ一貫して観測された現象は、他者認識と自己報告の超高連動である：

$$
\boxed{\text{Reader PredictionとSelf-Reportの強いBehavioral Coupling}}
$$

しかし、**行動レベルで類似している（Similar Behavior）からといって、内部表現を共有している（Shared Representation）ことや、同じ因果回路で実装されている（Shared Causal Implementation）ことの証明にはならない**。

したがって、次段階（V1）の研究課題は以下となる：

$$
\boxed{\text{行動レベルで類似するReader PredictionとSelf-Reportは、内部でも同じaffective representationを共有しているのか？}}
$$

$$
\boxed{\text{その情報は同じ因果サイト・因果経路で実際に利用されているのか？}}
$$

本論文における実験全体のロードマップは、以下の論理的展開を描く：

$$
\boxed{\text{EmoBank: Human Alignment}}
$$
$$
\big\downarrow
$$
$$
\boxed{\text{AIPsy-Affect: Controlled Affective Reactivity}}
$$
$$
\big\downarrow
$$
$$
\boxed{\text{V1: Reader vs. Self: Representation / Causality}}
$$

最終的に問うべき学術的核心は、以下の命題の真偽である：

$$
\boxed{\text{Behavioral Coupling} \stackrel{?}{\Longrightarrow} \text{Shared / Alignable Representation} \stackrel{?}{\Longrightarrow} \text{Shared Causal Implementation}}
$$



