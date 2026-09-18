emo v1–v3 の新規性評価
Deep Research レポート
既存研究との比較・査読者目線の厳格評価
2026-09-07
結論
研究テーマ自体は十分に論文化可能だが、「LLM内部に情動表現がある」「自己報告と内部状態を比較する」「activation steering/patchingで因果性を見る」だけでは、2026年時点では新規性は弱い。最も強い新規性候補は、post-training 前後で「情動表現そのもの」ではなく「内部表現→自己報告 readout の結合関係がどのように再マッピングされるか」を、paired base/instruct、cross-decoding、alignment、component-wise causal patchingで機構的に分解する点にある。
ただし、Martorell & Bianchi (2026) が「probe-defined emotive state と numeric self-report の causal informational coupling」をすでに正面から扱い、Anthropic (Sofroniew et al., 2026) が emotion representations・causal behavior・post-training変化まで扱っているため、論文の主張を単に「内部状態と自己報告は乖離する」に置くと、査読ではかなり危険である。
要素	新規性	競合の強さ	査読者評価
v1: 情動刺激→内部表現→自己報告	低〜中	非常に強い	単独では incremental
v2: Base/Instruct の表現変換・readout remapping	中〜高	中	主貢献にできる
v3: aligned cross-model causal patching	中〜高	中	設計次第で強い
v3: 自己報告＋行動の dual outcome	中	強い	Anthropicとの差分必須
全体: post-training による coupling 再編成	高になり得る	中〜強	最も defensible
1. 最重要の競合研究
1.1 Martorell & Bianchi (2026): Quantitative Introspection in Language Models
本研究に最も危険な競合。numeric self-report が probe-defined emotive internal state を追跡するかを検証し、introspection を「self-report と内部状態の causal informational coupling」と定義している。greedy self-report が少数値に collapse する一方、logit-based expected self-report で連続的な情報が回復し、activation steering により因果的結合も示す。LLaMA-3.2-3B-Instruct で Spearman ρ=0.40–0.76、モデルによって R²≈0.93 まで報告。TMLR under review。
•	重複: numeric self-report、logitベース評価、内部表現とのcoupling、steeringによる因果検証。
•	差分: 主眼は会話中の時間変化と introspective fidelity。Base→Instruct の post-training 比較や、表現空間の変換とreadout再マッピングの分解は中心ではない。
•	査読上の意味: 「自己報告は内部状態を反映する/しない」というRQだけでは新規性を主張できない。
1.2 Sofroniew et al. / Anthropic (2026): Emotion Concepts and their Function in a Large Language Model
171のemotion concept representationsをClaude Sonnet 4.5で同定し、preferences、reward hacking、blackmail、sycophancy等への因果効果をsteeringで示す。さらにbase/post-trainedモデルでemotion vector activationの変化も解析している。これは「情動表現＋因果行動＋post-training」の3点を既に押さえている。
•	特に重要: 著者ら自身が、base/post-trained比較では同じprobeを使い「emotion direction自体がpost-trainingでどう変わったかは測っていない」と明記している。
•	したがって、あなたのcross-decoding / Procrustes / Ridge alignmentで「方向・座標系そのものの変化」を測る部分は明確な空白を突ける。
•	一方、dual-outcome behaviorを「内部情動が行動を変える」とだけ主張するとAnthropicより弱い。
1.3 Keeman (2026): Whether, Not Which / AIPsy-Affect
AIPsy-Affectを用いた先行研究は、keyword-free clinical vignettes、matched neutral controls、base/instructモデル、linear probing、causal activation patching、knockout、representational geometryをすでに組み合わせている。しかもAIPsy-Affect自体が2026年4月に独立したdataset paperとして公開済み。
•	重複はかなり大きい。特に「AIPsy-Affect + probing + patching + base/instruct」は方法論上ほぼ既存。
•	新規性としてAIPsy-Affectの利用自体を前面に出してはいけない。むしろ既存batteryを使った新しいmechanistic questionとして位置付けるべき。
•	Whether, Not Which は affect reception と emotion categorization の解離が中心。あなたは internal representation と self-report readout の post-training-associated decoupling/remapping に焦点を移す必要がある。
1.4 Du et al. (COLM 2025): How Post-Training Reshapes LLMs
post-training前後の内部機構を比較するという上位レベルの発想は既存。knowledge、truthfulness、refusal、confidenceについて、表現方向の保存・変化・transferabilityを調べている。特にrefusal directionがbase/post-trainedで異なりforward transferが限定的という結果は、あなたの「post-trainingでreadout mappingが変わる」という議論に近い概念的先行研究になる。
2. 主要論文との比較
研究	内部情動	自己報告	因果介入	Base/Post	あなたとの差分
Di Palma ACL’25	Probe	弱	なし	一部	情動表現の存在は既知
Maheswaran EACL’26	Probe/方向	なし	限定	なし	cross-dataset表現が中心
Reichman ICLR’26	幾何	なし	介入	なし	emotion manifoldは既知
Sun et al. 2026	VA subspace	self-reportを軸学習に使用	steering	なし	VA+行動制御が既知
Keeman 2026	Probe/geometry	なし	patch/KO	あり	stimulus+方法が強く重複
Martorell 2026	Probe	numeric/logit	steering	ほぼなし	coupling自体が既知
Anthropic 2026	emotion vectors	preferences等	steering	あり	direction変化を未検証
本研究 v2/v3	VA/hidden	81候補尤度	patch/aligned patch	中心	couplingの再マッピング
3. v1 / v2 / v3 を査読者として採点
v1: 4/10（単独論文なら弱い）
情動刺激に対するVA自己報告、内部表現probe、ablation/patchingという組合せは、2026年の文献状況では新規な部品ではない。さらにAIPsy-Affectを使う場合、そのstimulus methodologyも既存研究の貢献である。v1は「現象発見・予備実験」としては有用だが、トップ会議の主貢献としては厳しい。
v2: 7/10（最も論文化しやすい核）
H1 Erasure / H2 Transformation / H3 Global Suppression / H4 Distributed Remapping を競合仮説として立て、within-model decodability、cross-decoding、Procrustes/Ridge alignment、mixed-effects coupling、component-wise patchingで識別する構成は、既存研究の単なるemotion probingから一段進んでいる。特に「post-training後も情報は保持されるが、表現幾何とself-report readoutの関係が変わる」という主張は比較的強い。
v3: 6.5–8/10（実験結果次第）
aligned cross-model patchingは強い。単にBase activationをInstructへ入れるのではなく、Base→Instructの表現変換を推定してからpatchし、それでもself-reportが回復する/しないことで「座標変換だけでは説明できるか」を因果的に試すからである。一方、dual-outcome behaviorはAnthropicのfunctional emotion→behavior causal effectsと近く、それ単体では新規性が高くない。
4. 査読で刺されるポイント
•	「self-reportを内部状態と呼んでいるだけでは？」: teacher-forced likelihoodは生成collapseを避けるが、それでも報告チャネルの確率であり主観状態ではない。用語をself-reported affective state / readoutに限定する必要がある。
•	「probeが読める ≠ モデルが使っている」: probingだけでは弱い。v3のcausal patchingを主結果に昇格させるべき。
•	「BaseとInstructはprompt format/tokenization contextが違うだけでは？」: identical token sequences、matched templates、same-position controlsが必須。
•	「Qwen2.5-1.5B一系統だけでは一般化不能」: 最低でもQwenの別scale + Llama/Gemmaのpaired base/instructを追加したい。
•	「post-trainingの何が原因か分からない」: Base vs InstructだけではSFT/RLHF/DPOの寄与を分離できない。中間checkpointが取れるモデル系列（例 OLMo系）でstage-wise解析すると大幅に強くなる。
•	「AIPsy-Affect依存」: 同じ著者・同じbattery由来の構造に過適合している可能性。EmoBank等の独立dataset、または別作成のkeyword-free matched stimuliで再現が必要。
•	「alignment mapの学習が単にsupervised transportしている」: held-out pairs、random alignment、orthogonal/linear/nonlinear比較、permutation controlが必要。
•	「Distributed Remappingという結論が強すぎる」: 単一MLP patchで戻らないだけでは分散機構の証明にならない。multi-site/path patching、mediation、additivity/interaction解析が欲しい。
5. 新規性を最大化する論文の置き方
避けるべき主張: 「LLMには感情表現がある」「自己報告と内部状態が乖離する」「情動表現をpatchすると出力が変わる」。これらは既にかなり埋まっている。
推奨する中心主張: 「Post-training does not simply erase or uniformly suppress affective information; it changes the geometry and, crucially, the causal readout mapping from affect-related representations to self-report.」
•	Contribution 1: paired base/post-trainedモデルで、情動情報の保持とself-report collapseを同時に定量化。
•	Contribution 2: cross-decoding/alignmentで representational transformation と information erasure を分離。
•	Contribution 3: layer/component-wise causal interventionsで global suppression と distributed readout remapping を識別。
•	Contribution 4: aligned cross-model patchingで「表現座標を合わせれば報告が回復するか」を直接検証。
•	Contribution 5: self-reportとbehavioral outcomeを分け、post-trainingが両readoutに同じ影響を与えるとは限らないことを示す。
6. 採択可能性を上げるための追加実験 優先順位
優先	追加実験	理由	効果
S	複数model familyのpaired Base/Instruct	一般化批判を潰す	非常に大
S	post-training stage別checkpoint	「post-training」の因果帰属を強化	非常に大
S	Martorell型logit self-report baselineとの直接比較	最重要競合との差分を実証	非常に大
A	multi-site/path patching	Distributed Remappingを実証	大
A	独立dataset replication	AIPsy依存を回避	大
A	alignment controls/permutation	transport artifactを排除	大
B	モデルサイズsweep	scaling claimを追加	中
7. 最終査読コメント（模擬）
Strengths. The paper asks a timely mechanistic question that is not fully answered by prior work: how post-training changes the mapping between affect-related internal representations and constrained self-report. The use of paired base/instruct models, cross-model alignment, and causal component replacement is potentially compelling.
Weaknesses. Several ingredients are already established: emotion representations, keyword-free clinical stimuli, activation patching, numeric/logit self-report, causal coupling between emotive states and reports, and post-training effects on emotion vectors. The paper therefore risks appearing as a recombination of recent work unless it demonstrates that readout remapping is a distinct, reproducible post-training phenomenon across model families and training stages.
Verdict. 現状なら Borderline / Weak Accept 相当。Qwen2.5-1.5B中心のままなら Weak Reject も十分あり得る。複数family・stage-wise post-training・multi-site causal evidenceまで揃えば、明確な mechanistic contribution として Strong Accept 圏に近づく。
8. 主要ソース
Martorell & Bianchi, Quantitative Introspection in Language Models (2026)
https://arxiv.org/abs/2603.18893
numeric/logit self-report と probe-defined emotive state の causal coupling。TMLR under review。
Sofroniew et al., Emotion Concepts and their Function in a Large Language Model (2026)
https://transformer-circuits.pub/2026/emotions/index.html
emotion vectors、causal behavior、base/post-training比較。
Keeman, Whether, Not Which (2026)
https://arxiv.org/abs/2603.22295
keyword-free clinical stimuli、base/instruct、probing、patching、knockout、geometry。
Keeman, AIPsy-Affect (2026)
https://arxiv.org/abs/2604.23719
480-item keyword-free matched clinical stimulus battery。
Du et al., How Post-Training Reshapes LLMs (COLM 2025)
https://arxiv.org/abs/2504.02904
post-training前後の表現方向・transferabilityのmechanistic比較。
Reichman et al., Emotions Where Art Thou (ICLR 2026)
https://proceedings.iclr.cc/paper_files/paper/2026/hash/51fa846f6b6b463144736fc3c3481f8c-Abstract-Conference.html
low-dimensional emotional manifoldとcausal intervention。
Maheswaran & Desarkar, A Unified View on Emotion Representation in LLMs (EACL 2026)
https://aclanthology.org/2026.eacl-long.165/
dataset横断の共通emotion representations。
Di Palma et al., LLaMAs Have Feelings Too (ACL 2025)
https://aclanthology.org/2025.acl-long.306/
layer-wise sentiment/emotion probing。
Sun et al., Valence-Arousal Subspace in LLMs (2026)
https://arxiv.org/abs/2604.03147
VA subspace、self-reported VAを用いた軸学習、steeringによるbehavior control。
Wu et al., Decoding and controlling emotion in LLMs (Computers in Human Behavior, 2026)
https://www.sciencedirect.com/science/article/pii/S0747563226001482
human-aligned VA geometryとSAE steering。
Singh et al., Can LLMs Introspect? A Reality Check (2026)
https://arxiv.org/abs/2605.26242
introspection claimsへの強い方法論的批判。

注: 本評価は2026-09-07時点で確認できた公開論文・プレプリント・査読中原稿を対象とする。プレプリント/査読中研究は最終採録状況が変わり得る。
