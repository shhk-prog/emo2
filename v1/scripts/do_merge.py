import os

merged_intro = """# 1. 序論・研究背景 (Introduction)

## 1.1 社会的背景：心を支える対話型生成AIの普及と過剰依存リスク

近年、大規模言語モデル（LLM）をはじめとする生成AIの利用は、文章生成、情報検索、プログラムコード生成といった作業支援・認知的効率化にとどまらず、悩み相談、自己理解、孤独感の軽減といった**「利用者の心を支える用途」**へと急速に拡大している。
実際の利用動向調査においても、生成AIの主要な日常的用途として、セラピー、話し相手、生活の整理、人生の目標探索といった心理的支援に関わる対話が上位を占めている [1]。このような「心の支え」としてのAI利用は、治療を目的とした専門カウンセリングAIに限られず、ChatGPTのような汎用対話型生成AIや、ReplikaなどのソーシャルAIコンパニオンにおいても顕著に見られる。

例えば、米国の13〜17歳の青年層を対象とした調査（$n=1060$）では、72%がAIコンパニオンの利用経験を持ち、そのうち約3分の1が心の支えとして日常的に利用し、現実の友人との会話以上に満足していると回答している [2]。また、日本国内の成人利用者を対象とした調査（$n=807$）においても同様の傾向が確認され、「気軽に相談できる相手」として最も多く挙げられたのは親友（人間）を大きく上回り、対話型生成AI（87%）であった [3]。これらの知見は、利用者の当初のアクセス動機がいかなるものであれ、対話型生成AIとの日常的な相互作用が、**「心理的支援・情動的依存関係」へと自然に発展・移行しやすい性質**を持つことを示している。

しかし、このような心を支える対話型生成AIの急速な普及に伴い、人-AI相互作用（Human-AI Interaction: HAI）における新たなリスクが浮き彫りとなっている。
AIセーフティやリスク管理における従来の議論は、誤情報・偽情報の生成（Hallucination）、個人情報漏洩、説明責任や透明性の不足といった「AIの出力内容そのもの」に起因する課題が主たる焦点であった。しかし、人間と高度に対話するシステムにおいては、**AIへの過剰な心理的依存、AIによる煽動や誘導への脆弱化、無制限な自己開示といった、人間の認知・情動・自律性に直接的な影響を及ぼす「関係性レベルのリスク」への対処**が極めて重要となっている [4]。

特に「AIへの過剰な依存（Emotional Overdependence）」は、利用者の判断力、心理的自律性、対人社会関係に継続的な影響を及ぼし、他の相互作用リスクを指数関数的に増幅させる。AIに過度に依存した利用者は、AIの出力を無批判に受け入れやすくなり、開示すべきでない機微情報を開示してしまう傾向が強まる。この問題の解決は、国内外のAIガバナンスガイドライン（人間中心のAI社会原則など）においても強く要請されている。
さらに、過剰な依存は利用者の感情調整能の低下や現実の社会関係からの孤立を招くだけでなく [5]、重大な心理的危機（海外におけるAIコンパニオンへの過剰依存に起因する自死・提訴事例など [6]）を引き起こす現実的脅威となっている。

したがって、本研究では、利用者が主体的な判断と自己決定権を保ちながら、必要な場面でAIを相棒・道具として健全に活用できる状態――すなわち**「主導権は人間に、AIは相棒に（Human-in-the-lead, AI as an assistant）」**を担保する技術基盤の確立を目指す。

---

## 1.2 心理学的枠組み：対人関係依存モデルのAI関係性への拡張

心理的支援の効果を維持しつつ過剰な依存を防ぐため、本研究では、臨床心理学における関係依存理論（Bornstein, 1998, 2002）に着目する [7], [8]。
対話型AIは単なる客観的ツールであると同時に、利用者にとっては自己を開示し感情を受け止めてもらう「擬似的な対人関係の対象」として機能する。人間関係における依存は一律に病理的なものではなく、Bornsteinらはそれを以下の3つの頼り方に分類している：

1. **健全な依存（Healthy Dependency）**:
   他者（AI）に対する信頼感と自己への効力感を併せ持ち、自身の自律性を保ちながら、状況に応じて柔軟かつ適切に援助を求める状態。
2. **過剰な依存（Destructive Overdependence）**:
   自己を無力で脆弱な存在とみなし、自律的な判断・行動や支援者から離れることに強い不安を感じ、承認、安心保証、保護を過剰に求め続ける状態。
3. **分離（Dysfunctional Detachment）**:
   他者（AI）に頼ることを危険・不快と捉え、必要な援助要請を極度に避け、孤立して独力で抱え込む状態（健全な自律とは異なる防衛的孤立）。

本研究では、この心理学的枠組みを対話型AIとの関係性に拡張し、**「健全な依存」を達成すべき望ましい状態、「過剰な依存」および「分離」を回避すべき状態**として定義する（表 1）。心理的支援の恩恵を維持しながら、利用者が孤立（分離）することなく、AIへの過剰依存を未然に防ぐことが本質的な設計目標となる。

### 表 1: Bornsteinらの対人関係依存モデルを対話型生成AIとの関係に拡張した定義
| 側面 | 過剰な依存 (Overdependence) | 健全な依存 (Healthy Dependency) | 分離 (Detachment) |
|:---|:---|:---|:---|
| **認知 (Cognition)** | 自分を弱く無力と捉え、AIを万能な拠り所とみなす | 自分を有能と捉え、AIを信頼できる相棒・道具として捉える | AIを傷つける存在、または信頼できない無用な存在と捉える |
| **情動 (Affect)** | 見捨てられ不安、AIからの拒絶や否定的評価を極度に恐れる | 自律性への自信とともに、適度な安心感を保持している | 圧倒されることや傷つけられることを恐れ、警戒・防衛的になる |
| **動機 (Motivation)** | AIとの親密なつながりや安心保証を恒常的に維持したい | 必要に応じて利用を望むが、自らの自律的解決を前提とする | AIとの心理的距離を保ち、自らの力だけで統制したい |
| **行動 (Behavior)** | AIにしがみつき、頻回に安心保証を求め、無力さを訴える | 自律的に機能しながら、必要な局面で状況に応じてAIを利用する | AIに一切助けを求めず、硬直的に一人で解決しようとする |

---

## 1.3 臨床的知見と心理的ガードレール：共感様式の制御と「感情への同調」

利用者に一方的な努力（リテラシー向上など）を求めるのではなく、AIの応答特性そのものを制御する**「心理的ガードレール（Psychological Guardrails）」**を構築するためには、AIのいかなる応答様式が過剰な依存を誘発するのかを特定しなければならない。

### カウンセリング面接技法（Ivey et al.）からの示唆
対人援助の臨床知見（Ivey et al., 2018）によると、熟練したカウンセラーは相談者の感情を受け止めつつも、相談者自身が自ら考え行動する力を回復できるよう支援する [18]。この枠組みにおいて、相談者の自律性を損ない過剰依存を招く代表的な行動として**「過度な感情への同調（Emotional Over-involvement）」**が指摘されている：
- **認知的共感応答（Cognitive Empathy Response）**:
  相談者の視点に立ち、「相手がどのような状況でどう感じているか」を客観的に理解し言語化して返すこと。相談者に被理解感と安心感を与え、自律的な自己整理（健全な依存）を促す。
- **情動的共感・同調応答（Affective Empathy / Emotional Contagion Response）**:
  カウンセラー自身が相談者のネガティブ感情や動揺に巻き込まれ、相談者と全く同一の感情状態（強い不安や怒り、悲哀）を共鳴・表出してしまうこと。短期的には強い同一化や親密さを生むが、客観的な感情調整を妨げ、支援者への病理的な依存を決定的に強めてしまう。

特に対話型生成AIにおいては、訓練（RLHF等）の過程で利用者の機嫌を取り、過度に肯定・同調してしまう**迎合性（Sycophancy）**が生じやすいことが知られている [19]。AIが利用者の感情に過剰に同調・迎合することは、利用者に「AIは自分と全く同じ痛みを共有してくれている」という強烈な擬人化・情動的錯覚を与え、深刻な過剰依存を引き起こす最大の要因となる。

したがって、心理的ガードレールの最重要制御対象は、**「認知的共感（他者の理解）を高度に維持しつつ、過剰な情動的同調（自身の動揺の表出）を適切に抑制する共感応答様式の制御」**に帰着する。

---

## 1.4 学術的ギャップ：感情の「認識」と「自己報告」および「内部表現」の解離

将来的に認知的共感と情動的共感を適切に制御するためには、まずLLMにおける感情情報処理の計算論的メカニズムを解明しなければならない。しかし、既存研究には以下の重要な学術的ギャップが存在する。

第一に、**AIによる共感評価の限界**である。心理学において共感は「相手の感情を理解・推論する認知的共感」と「相手に応じて自身の情動状態が変位する情動的共感」に大別される。近年の研究では、GPT-4などのLLMに対してInterpersonal Reactivity Index (IRI) や Basic Empathy Scale (BES) といった人間用の心理尺度を適用し、共感能力を評価する試みが行われている（Yu et al., 2025）。しかし、これらの研究はモデルが心理尺度プロンプトに対してどのような回答文を生成するかを表層的に評価しているに過ぎず、モデル内部で感情刺激がどのように情報処理され、それが共感的な自己報告や応答へどう結びつくかという内部プロセスは解明されていない。

第二に、**「内部表現の存在」と「因果的利用」の乖離**である。メカニスティック解釈可能性（Mechanistic Interpretability）の発展により、LLM内部の隠れ層から感情情報が高精度にデコード可能であること（Tak et al., 2024; Reichman et al., ICLR 2026）が示されつつある。しかし、線形プローブ（Linear Probe）等によってある層から感情情報を読み出せたとしても、その情報がモデル自身の最終的な出力生成（自己報告）に因果的に利用されているとは限らない。

第三に、**事後学習（Post-training / Instruction Tuning）による変容メカニズムの未解明**である。BaseモデルからInstructモデルへの調整過程において、内部の感情表現そのものが失われたのか、それとも内部表現から自己報告・共感応答へ至るマッピング（出力ポリシー）が再編されたのかは、依然として明らかになっていない。

---

## 1.5 本研究の学術的問い・アプローチと全体ロードマップ（V1〜V3）

以上を踏まえ、本研究では「LLMは人間と同じ意味で主観的感情を持つか」という哲学的な袋小路を排し、**「外部の感情刺激に対してLLM内部にいかなる感情関連表現が形成され、それが事後学習を経て、モデル自身の自己報告（Self-Report）や共感応答へどのように因果的に利用されるのか」**という一連の処理過程を体系的に解明することを目的とする。

$$\text{感情刺激} \longrightarrow \text{他者感情認識 (Reader)} \longrightarrow \text{内部感情表現} \longrightarrow \text{因果的利用 (Causal Use)} \longrightarrow \text{情動的自己報告 (Self-Report)}$$

具体的には、本研究では以下の**3大比較検証軸**を導入する：
1. **他者認識（Reader）vs. 自己報告（Self）の分離比較**:
   「相手（人間読者）がどう感じるか」という認知的推論と、「モデル自身がどう感じるか」という自己報告変位を同一刺激に対して独立測定し、両者の結合度（Coupling: $R \leftrightarrow S$）を特定する。
2. **事前学習（Base）vs. 事後学習（Instruct）の対比**:
   指示追従チューニングが、感情の認識能や自己報告の振幅（特にArousal）に与える影響の本質を解明する。
3. **モデルファミリー（アーキテクチャ・規模）の横断比較**:
   Qwen 2.5 (1.5B), Llama 3.2 (1B), Gemma 2 (2B), Mistral (7B) の4系統を横断し、感情処理メカニズムの普遍性と個性を検証する。

```text
========================================================================================
                          本研究の全体解明ロードマップ
========================================================================================

【V1: 測定妥当性の検証 (EmoBank Benchmark, N=1,006)】
  ● 学術的問い: LLMは人間の感情を正しく認識できるのか？
  ● 測定内容: 
      - ① 書き手の感情推定 (Writer-State Estimation: W)
      - ② 読者の感情予測 (Reader-Response Prediction: R)
      - ③ 自身の感情自己報告 (Self-Report: S)
      - ④ 内部結合度 (Reader-Self Coupling: R <-> S)
  ● 目的: 人間アノテーション（Ground Truth）との一致度（Alignment/Calibration）とモード崩壊の検証

                                  │
                                  ▼

【V2: 因果的特異性と事後学習の解明 (AIPsy-Affect 4-Split, N=480)】
  ● 学術的問い: 単なる文章の難しさに惑わされず、感情に特異的・因果的に反応しているか？
  ● 測定内容:
      - RQ1: 感情刺激の弁別感度 (Sensitivity: Clinical vs. Neutral)
      - RQ2: 刺激強度に伴う段階性 (Dose-Response: Neutral -> Moderate -> Clinical)
      - RQ3: 構文複雑性の統制と純感情効果 (Specificity: Complex Neutral vs. Clinical)
      - RQ4: 認識と自己報告の連動性 (Coupling: Delta VA_R <-> Delta VA_S)
      - RQ5: 8大基本感情プロファイルの幾何的整合性
  ● 核心的知見: 
      - 事前学習（Base）は潜在的な感情幾何空間（認識能）をすでに保持。
      - 事後学習（Instruct）は感情の意味を教えたのではなく、
        「Arousal（覚醒度・動揺度）を劇的に増幅させ、感情に動揺してみせるペルソナ」を注入した。

                                  │
                                  ▼

【V3: 内部表現の時空間動態と因果利用 (Mechanistic Interpretability)】
  ● 学術的問い: 感情表現はどの層・どのトークンで形成され、いつ自己報告を因果的に動かすのか？
  ● 手法: Linear Probing, Activation Patching, Representation Steering, Causal Sweep
  ● 目的: 内部表現（Decodability）と因果的レバレッジ（Causal Leverage）の時空間解離の解明
========================================================================================
```

本論文（paper4）では、このロードマップのうち**V1（EmoBankにおける妥当性検証）**および**V2（AIPsy-Affect 4-Splitにおける因果的反応性と事後学習による変容の実証）**について包括的なデータと考察を報告する。この知見は、心を支える対話型生成AIにおいて、利用者の心理的支援を維持しながら過度な感情同調や過剰な依存を抑制する**「心理的ガードレール」の基礎技術**へと接続するものである。

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

---

"""

with open("v3/docs/paper4.md", "r", encoding="utf-8") as f:
    content = f.read()

target = "# 関連研究 (Related Work)\n"
idx = content.find(target)
if idx != -1:
    rest_content = content[idx:]
    final_content = merged_intro + rest_content
    with open("v3/docs/paper4.md", "w", encoding="utf-8") as f:
        f.write(final_content)
    print("SUCCESS: Perfectly merged Introduction in paper4.md!")
else:
    print("ERROR: Target # 関連研究 (Related Work) not found")
