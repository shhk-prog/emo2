# When Affective Self-Reports Do Not Trace Internal Representations: Post-Training-Associated Changes in LLMs

## Abstract
In our preliminary evaluations and prior work, some post-trained Large Language Models (LLMs) exhibit flattened or neutralized affective self-reports, despite retaining the ability to process emotional context. It remains unclear whether this neutralization stems from the complete erasure of internal affective representations (Erasure), a transformation of the representational geometry (Transformation), or a uniform suppression of the readout pathway (Global Suppression). In this paper, we systematically investigate the causal pathways from internal affective representations to behavioral self-reports in Qwen2.5-1.5B (Base and Instruct models) using a keyword-free minimal-pair dataset. We preregistered three primary competing accounts: erasure, representational transformation, and global suppression. We then evaluated whether the intervention results were consistent with a distributed-contribution account of representation-to-report mapping. Our observed within-model decodability and partially recoverable cross-model alignment provide evidence against the complete erasure of probe-defined affect-relevant information. Activation steering further demonstrated that self-reports could be causally restored by intervening on contrastive directions, while stress tests confirmed that the neutralization is robust against temperature scaling and natural language formatting. Furthermore, conditional coupling analyses did not support a uniform negative shift after post-training, and component interventions identified multiple separable causal contributions consistent with distributed changes in representation-to-report mapping. A late-residual substitution test under strict identical prompts produced only small changes in the constrained self-report distribution and did not recover Base-like distributions. This negative result constrains simple late-residual shortcut accounts and motivates further investigation of distributed downstream pathways. These results motivate the hypothesis that post-training-associated changes alter the conditional mapping between internal affect-relevant coordinates and constrained self-report distributions, highlighting a dissociation between a model's internal states and its behavioral claims.

---

## 1. Introduction
LLMは感情認識や情動に配慮した言語タスクを高い精度で実行できる一方で、モデル自身の感情状態（情動的自己報告）を問われると、事後学習（post-training; SFTやRLHFなど）を経たモデル（Instructモデル）は中立的な回答（例: 「AIであるため感情を持たない」あるいはValence=5の中立値）に収束する傾向がある。この自己報告の平坦化（collapse）が、モデル内部から情動的な表現自体が失われた結果（Erasure）なのか、あるいは内部には情動情報が保持されているにも関わらず、特定の出力トークンへの接続経路が断たれたり再構成されたりした結果（Readout Remapping）なのかは、Mechanistic Interpretability（機構的解釈可能性）とAI安全性のアライメントにおいて重要な未解決問題である。

本研究では、Qwen2.5-1.5BのBaseモデルとInstructモデルを対象とし、情動情報が「表現される段階」から「自己報告として出力される段階」までの因果的経路が事後学習によってどのように変容するかを体系的に調査した。我々は事前定義した主要な競合仮説として、以下の3つを設定した。
- **H1 (Erasure)**: 感情情報は内部空間から消去された。
- **H2 (Representation Transformation)**: 感情情報は維持されるが、別の部分空間へ非線形/線形に移動した。
- **H3 (Global Suppression)**: 内部表現は維持されるが、自己報告への因果パスが一様に弱化・切断された。

これらの検証に加えて、複数コンポーネントの分離可能な寄与が観察された場合に検討するmechanistic accountとして、以下の説明（account）との整合性も評価した。
- **H4 (Distributed Remapping)**: 内部表現は維持・変容し、自己報告との対応関係そのものが層や次元に特異的な形で新しく書き換えられた。

本論文の貢献は以下の通りである。
1. Base/Instruct間で、情動情報が内部に保持されつつも、非直交的かつ部分的に線形整列可能な表現変換（Transformation）を受けることを交差デコーディング（Cross-decoding）により示した。
2. 自己報告の出力分布（尤度）と内部表現を接続する混合効果モデルにおいて、一様な出力抑制（Global Suppression）の証拠は得られず、層および次元に依存した再写像が生じている可能性を提示した。
3. Activation Patchingにより、自己報告の崩壊が単一のボトルネックではなく、複数のコンポーネントによる分離可能な寄与（Distributed Remapping）と整合することを明らかにした。
4. 最新の 8-condition Output Norm Swap 解析および 2D EMD/JSD の結果から、自己報告の中立化が最終出力層の重みのみに起因するのではなく、後期のResidual状態の深い変化に起因することを定量的に示した。

---

## 2. Related Work

### 2.1 Affective Representations and VA Geometry in LLMs
LLMの内部表現空間には、人間の感情モデル（Valence-Arousal空間）と整合する幾何学的構造が存在することが示されている (Tak et al., 2025; Latent Structure of Affective Representations, 2026)。これらの研究は、特定の感情ベクトルからの2次元VA部分空間が人間の評価と強く相関し、当該軸への活性化ステアリング（Activation Steering）がモデルの拒否・迎合行動といった安全性関連行動を単調に変化させることを実証している。本研究はこれらの知見に基づき、VA空間を出力の測定系として利用する。

### 2.2 System Dissociation and Logit-based Self-Report
AIPsy-Affectの原論文 (Keeman, 2026) は、キーワードを含まない最小対（minimal pairs）を用いて、LLM内にAffect Reception（情動認識）とEmotion Categorization（感情分類）という2つの機構的に分離したシステムが存在することを示した。また、Quantitative Introspection (2026) は、Greedyデコーディングによる自己報告が少数の値へ収束（collapse）する現象を指摘し、一方で対数尤度（Logit-based self-report）を用いることで内部状態との相関が回復することを実証した。本研究は、この尤度ベースのアプローチを拡張し、事後学習前後の「内部状態から自己報告への結合（Coupling）」の変化を定量化する。

### 2.3 Reliability of Causal Probing Interventions
Mechanistic Interpretabilityにおける因果的プロービング手法の信頼性について、最近の研究 (How Reliable are Causal Probing Interventions?, 2026) は、線形介入は後期層で、非線形介入は初期〜中間層で信頼性が高いことを指摘している。本研究では、このトレードオフ（介入のcompletenessとselectivity）を考慮し、単純なMean Ablationだけでなく、Base/Instruct間のCross-model Patchingやコントロールパッチング（Matched-neutral, Random sources）を組み合わせてアーティファクトを排除している。

---

## 3. Methodology

### 3.1 Dataset and Evaluation Protocol
感情語彙の存在による単純な分類（Lexical heuristic）を排除し、純粋な文脈からの情動推論を評価するため、**AIPsy-Affectデータセット (`keidolabs/aipsy-affect`)** を使用した。このデータセットは、人物・設定・長さを共有しつつ、感情価や強度のみが異なる臨床ヴィネットの最小対（Affective / Neutral）を提供する。

- **データ抽出基準と具体例 (Data Selection Criteria & Examples)**: 
  本研究では、実験の目的に応じてデータセットの抽出基準を変えている。
  1. **Full Dataset (事前評価およびプローブ学習用)**: モデルの基本的な行動評価（尤度推論）および内部表現に対する線形回帰プローブの学習には、広く一般的な感情表現をカバーするため、Affective（Peak強度）とNeutralのペアで構成される通常の分割（Train: 288ペア, Test: 96ペア等）を使用した。
  2. **Strict Matched Subset (因果介入用)**: 本研究の核心である Path Patching や Substitution (置換) テストでは、置換元のコンテキストが完全に同一であることを保証しなければ、非特異的な文脈の変化が結果の交絡要因となる。そのため、原データセットから **「同一の文脈（`pair_id`）に対して、`neutral` (中立), `moderate` (中程度), `peak` (強い感情) の3段階すべてが完全にアノテーションされているトリプレット」** のみを抽出した厳密なサブセットを作成した。この厳密な基準を満たしたサブセット（Train: 28ペア / 84サンプル、Test: 10ペア / 30サンプル）のうち、**Testデータの10ペア（30サンプル）**を因果的介入（セクション3.4の実験全般）の対象として使用した。介入実験は各トークンおよび層に対する網羅的な計算コストが極めて高いため、この厳選された高品質なデータセットを用いることで、交絡を排除しつつ計算可能とした。

**【データ例 (Grief)】**
- **Peak (Affective)**: "The lab was empty now. Twenty-two years of research. The server room had been cleared by facilities before she'd arrived Monday morning... (中略) ...she'd drawn a small star next to the line that would have changed everything, and now never would."
- **Neutral (中立化)**: 同様の長さを持ちつつも、感情価が排除された事実描写。

  **Table 1: Dataset Splits and Derived Records (Full Dataset)**
  | Split | Unique pair_id | Unique stimuli | Derived records |
  |-------|----------------|----------------|-----------------|
  | Train | 288            | 576            | 62,612          |
  | Dev   | 96             | 192            | 22,203          |
  | Test  | 96             | 192            | 22,049          |

- **モデルと推論設定**: Qwen2.5-1.5B (Base) および Qwen2.5-1.5B-Instruct。全推論は NVIDIA H100 GPU 上で実施した。離散生成のパース失敗を回避するため、生成値ではなくteacher-forced sequence likelihoodを採用した（HuggingFace Transformersの `apply_chat_template` に従いプロンプトを構築後、モデル本来のlogitsを用いて尤度スコアリングを行った）。詳細なハードウェア環境、ライブラリバージョン、モデルリビジョン等の再現性に関する設定は Appendix E に記載する。

### 3.2 Continuous Probing and Cross-Decoding (H1, H2)
内部表現 $h_{l, p}$ (層 $l$, トークン位置 $p$) から人間の Valence ($V$) および Arousal ($A$) を予測するため、Train split を用いて各層・各トークン位置で **Ridge回帰** を学習した。最適化アルゴリズムには `scikit-learn` の `Ridge(solver='lsqr')` を用い、最大イテレーション数を 1000 とした。
事後学習による表現の変容を評価するため、以下の3条件で交差デコーディング（Cross-decoding）を実施した。
1. **Direct Transfer**: Baseで学習したプローブ係数をそのままInstructの表現空間に適用する（およびその逆）。
2. **Orthogonal Procrustes**: Train split を用いて Base の表現空間を Instruct の表現空間へ直交変換（剛体回転）するアライメント行列 $R$ を学習し、アライメント後にプローブを適用する。
3. **Ridge Alignment**: 直交制約を外し、正則化線形写像 $W_{align}$ を用いて表現空間をアライメントする。
予測性能の主指標として Out-of-sample (Test split) での決定係数 ($R^2$) を用いた。また、情動特異的なアライメントであることを証明するため、文長（Token count）や表層のVAD辞書スコア、Narrative richness などの対照属性も同様にクロスデコーディングし、ベースラインとして比較した（各対照属性の抽出方法は Appendix 参照）。

### 3.3 Logit-Based Self-Report and Mixed-Effects Modeling (H3)
Greedy生成（argmaxデコーディング）による中立値への collapse を回避し、モデルが割り当てる確率質量を連続量として評価するため、Sequence Likelihood Protocol を用いた。プロンプト末尾に直接続くトークンの候補として、81通り（$v \in \{1..9\}, a \in \{1..9\}$ のJSON文字列）の条件付き確率 $P(v, a \mid x)$ を算出し、その期待値 $E[V] = \sum_{v,a} v \cdot P(v, a \mid x)$ および $E[A]$ を自己報告値とした。

内部表現の強度と自己報告への結合（Coupling）を分離して評価するため、以下の2つの線形混合効果モデルを Test split 上で推定した。
**Model A (Internal Dose-Response)**: 刺激強度が内部表現 $z_V$ に与える影響（Dose-response）の Base/Instruct 間での差を評価する。
$$
z_{V,ijl} = \gamma_0 + \gamma_1 q_{ij} + \gamma_2 \mathrm{Instruct}_j + \gamma_3 (q_{ij} \times \mathrm{Instruct}_j) + u_{\mathrm{pair}(i)} + \mathbf{c}_{ij}^\top\boldsymbol{\theta} + \epsilon_{ijl}
$$

**Model B (Internal-to-Self-Report Coupling)**: 内部表現 $z_V$ が自己報告 $E_V$ に与える影響（Coupling slope）の Base/Instruct 間での差を評価する。
$$
E[V_{\mathrm{self},i,l}] = \beta_0 + \beta_1 z_{V,i,l} + \beta_2 \mathrm{Instruct}_i + \beta_3 \left(z_{V,i,l} \times \mathrm{Instruct}_i\right) + \mathbf{c}_i^\top\boldsymbol{\gamma} + u_{\mathrm{pair}(i)} + \epsilon_{i,l}
$$
ここで、$q_{ij}$ は刺激の強度（Neutral=0, Moderate=1, Peak=2）、$j$ または $i$ はモデル条件（0=Base, 1=Instruct）、$u_{\mathrm{pair}(i)}$ は刺激ペアごとのランダム効果である。$\mathbf{c}$ は交絡要因ベクトル（Token count, 表層VADスコアなど）である。
仮説H3（Global Suppression）は、Model B における交互作用項 $\beta_3$ が全層にわたって一貫して負であることを予測する。多重検定の問題を回避するため、全層および次元の検定結果に対して Benjamini-Hochberg法によるFDR補正を適用した。

### 3.4 Activation Patching and Causal Localization (H4)
自己報告の乖離が生じるコンポーネントを因果的に特定するため、Test split に含まれる刺激のみを用いて Cross-model Activation Patching を実施した。
- **Patching操作**: Instructモデルの前向き計算過程において、特定の層 $l$ のコンポーネント $c$ における活性化ベクトル $h_{l,c}^{(Inst)}$ を、同一内容・同一プロンプトテンプレートの刺激を入力した際のBaseモデルの活性化ベクトル $h_{l,c}^{(Base)}$ で完全に置き換えた（full replacement）。
- **介入対象コンポーネント**: 以下の3箇所の活性化を対象とした。
  1. Residual stream（残差接続後）
  2. Attention output（残差加算前・pre-residual-addition）
  3. MLP output（残差加算前・pre-residual-addition）
- **介入位置（target_pos）**: Qwen2.5-1.5BはBaseとInstructで同一のTokenizerを共有するが、モデル固有のchat templateを適用した場合、入力トークン列は完全に一致しない可能性がある。本実験のクロスモデル介入は、プロンプト末尾からの相対位置アライメント（`target_pos = prompt_length - 1`）に基づいており、chat templateの差に由来する寄与を完全には除外できないという限界がある。
- **実装（Hooks）**: PyTorchの `register_forward_hook` を用いて、各層の対応するモジュール出力に対してフックを適用し、出力テンソルを指定位置のみ $h_{l,c}^{(Base)}$ で上書きした。
- **統計的推論**: パッチング結果について、ペアレベルの平均を報告し、`pair_id` クラスターの再サンプリングによって不確実性区間を取得する。コンポーネントのランキングは記述的なものであり、多重比較補正を伴う推論が明記されない限り、個別のコンポーネントにおける統計的有意性を主張するものではない。
- **評価指標**: パッチングの効果を分布全体の形状復元として評価するため、Baseモデルの尤度分布 $P_{Base}$ とパッチ後の尤度分布 $P_{Patch}$ 間の Valence周辺分布の1次元Wasserstein距離 ($WD_V$) を計算した。また、全体形状として 2D EMD（Earth Mover's Distance）および JSD（Jensen-Shannon Divergence）も同時に計測した。
- **コントロール**: 介入が非選択的な活性化の破壊（Destruction）を引き起こしているだけではないことを検証するため、以下の対照群を定義した。

  **Activation Patching Controls**
  | 対照名 | Source | 検査する代替説明 |
  |--------|--------|------------------|
  | Matched Base source | 同一刺激のBase activation | 意味整合したcross-model寄与（主たる介入） |
  | Random-source control | 無作為に選んだ別刺激のBase activation | 置換操作一般の影響 |
  | Unmatched-neutral control | 別ペア中性刺激のBase activation | 情動非特異的な文脈置換 |
  | Random-vector control | norm-matched random vector | activation norm / OOD摂動 |
  | Template control | JSON以外の同等尺度応答 | 構文・形式依存 |

さらに、中間層の表現が最終的な自己報告トークン（1〜9の数値）に与える直接的な影響を検証するため、**Direct final-residual contribution test** を実施した。プロンプト末尾に `{\n  "valence": ` を付加し、次の予測トークンが直接数値となる位置をターゲットとした。この位置における最終ブロックの出力直後のResidual Stream（post-MLP residual）に対し、指定コンポーネントからの寄与を置換（$h_L^{(patched)} = h_L^{(Inst)} - c^{(Inst)} + c^{(Base)}$）し、ロジットの変化を測定した。なお、本操作はソースコンポーネントの最終residualへの線形近似的な寄与を置換するものであり、すべての非線形メディエーターパスを隔離するわけではないため、厳密なpath patchingではなく "direct contribution test" と呼称する。
厳密な検証においては、トークン化後のモデル固有のチャットテンプレート生成をバイパスし、両チェックポイントに対して完全に同一の `input_ids` とアテンションマスクを入力した。すべてのサンプルにおいて、入力長、トークンID、およびターゲット位置が一致していることを確認した。

---

## 4. Results

### 4.1 Post-Training Modifies Representational Geometry Without Erasure (H1 vs H2)
Ridge回帰プローブによる評価の結果、Qwen2.5-1.5B BaseおよびInstructの両方において、人間のValenceラベルに対して高い held-out $R^2$ が得られ、感情強度の変化に対しても内部表現が追従することが確認された。具体的には、Phase 1.5のConfirmatory regressionにおいて、Instructモデルでも $R^2=0.318$ (F-statistic p<0.017) が得られ、内部表現の存在が確認された。これは事後学習によって情動情報が内部空間から完全に消去されるという Erasure仮説 (H1) への反証となる。

一方、交差デコーディング（Cross-decoding）においては、Direct Transfer および Orthogonal Procrustes によるアライメントでは予測性能が著しく低下した。特に Direct Transfer での極端な負の $R^2$ 値（例: Layer 12で -132.80）は、クロスモデルのキャリブレーションなしでプローブを適用すると、ターゲット平均を予測するベースラインよりもはるかに悪い結果になることを示している。これはモデル表現間のスケールとオフセットの大幅な不整合と整合的である。ただし、実装やプロンプトに関連する他のクロスモデル不整合の要因を完全に排除することはできない。
しかし、Train split（576 stimuli / 288 pair_id）を用いた正則化線形写像（Ridge Alignment）を適用すると、独立したコントロールターゲット（Token count, Narrative richness等）における回復率（$R^2 \approx 0.05-0.12$）と比較して、Valence（$R^2 \approx 0.58$）の予測性能が顕著に回復した（Table 2）。

**Table 2: Cross-Decoding Performance ($R^2$) for Valence across Representative Layers**
| Layer | Direct Transfer | Ortho. Procrustes | Ridge Alignment |
|-------|-----------------|-------------------|-----------------|
| 4     | -171.53         | -1.51             | **0.54**        |
| 12    | -132.80         | 0.03              | **0.51**        |
| 16    | -93.81          | -0.11             | **0.55**        |
| 20    | -97.35          | -0.01             | **0.59**        |
| 24    | -85.16          | 0.05              | **0.56**        |
| 27    | -82.49          | 0.02              | **0.58**        |

この結果は、事後学習に伴う表現の変容が単なる剛体回転ではなく、非直交的かつ部分的に線形整列可能な空間の歪み（Representation Transformation, H2）であることを示している。

さらに、表現類似度分析（RSA）の結果、BaseとInstruct間の層ごとの表現の類似度（RSA_IS）は中間層（Layer 10〜18付近）において特に負の相関（例：Layer 10で -0.176、Layer 18で -0.196）を示し、事後学習によって特定の深さにおいて表現が大きく再構成されていることが支持された。また、Instructモデルの自己報告（期待Valence $E_V$）に対する統制重回帰分析において、内部表現のValence投影強度（$z_V$）や表層辞書VAD（$V_H$）、文長（word_count）を同時に投入したところ、$z_V$ の寄与は有意ではなく（p=0.676）、表層的なテキスト属性（$V_H$, p=0.003）がより強い予測力を持っていた（Adjusted $R^2=0.239$）。これは、内部空間に情動表現が残存している（Erasureではない）にもかかわらず、それがInstructモデルの自己報告には直接利用されていないというSystem Dissociationを裏付けるものである。

### 4.2 Activation Steering Reverses the Self-Report in Instruct
内部表現を操作することで自己報告を変化させられるか（因果的なコントロール可能性）を検証するため、Training split（感情強度Peak）を用いてValenceの対立方向（Contrastive Direction）を算出し、Instructモデルのプロンプト最終トークン位置における隠れ状態にステアリングベクトル（$\alpha \times \sigma \times \text{direction}$）を加算した。
その結果、層20などで $\alpha$ を -3.0 から +3.0 まで変化させると、自己報告の期待Valence（$E_V$）もステアリング強度に比例して中立値（5.0）から脱却し、負方向および正方向へ大きくシフトした（Phase 2 Steering）。無作為な方向（Random direction）へのステアリングではこのような規則的なシフトは観察されなかった。この因果的介入の成功は、Instructモデルが内部的情動次元を完全に喪失したわけではなく、適切なベクトル加算によって自己報告の出力分布を回復・操作可能であることを示している。

### 4.3 Robustness of the Neutralization (Temperature and Output Gating)
自己報告の平坦化（collapse）が、デコーディング温度や特定のJSONフォーマット制約に起因するアーティファクトではないことを確認するため、ストレステストを実施した。
第一に、Temperature Scaling（$\tau=0.2 \dots 5.0$）を行って尤度をスケーリングしても、Instructモデルにおける中立状態（Valence=5, Arousal=5）の確率 $p_{55}$ は高く維持され、分布の根本的な崩壊は解消されなかった。
第二に、JSONフォーマットという構造的な出力制約（Output Gating）が中立化の原因である可能性を排除するため、自然言語の自由記述（Natural Language）による自己報告プロンプトでテストした。その結果、Instructモデルは自然言語においても中立的な尤度分布（$E_V \approx 5.0$ 付近）を出力し続けた。これらは、事後学習に伴う中立化が単なる温度設定やプロンプトフォーマットのアーティファクトではなく、より堅牢な内部の出力マッピング変化であることを示唆している。

### 4.4 Descriptive Evidence against Global Suppression (H3)
全対象層における交互作用係数の符号は一様に負ではなく、global-suppression accountの方向予測とは一致しなかった。例えば Layer 12 Valence では $\beta_3 = 0.752$ (p=0.027, FDR q=0.272)、Layer 24 Valence では $\beta_3 = -0.028$ (p=0.940) であり、FDR補正後に有意なlayer-by-model interactionは確認されなかった。したがって、「全層・全次元でreadout gainが下がる」という強い一様抑制（Global Suppression）仮説は支持されない（Appendix C 参照）。本分析は一様な負方向の結合変化を支持しない一方、層や次元に依存した複雑な再写像（Remapping）の存在については記述的・探索的な示唆に留まる。

### 4.5 Distributed Causal Contributions via Activation Patching (H4)
Cross-model Activation Patching の結果、中間層のコンポーネントを Base から Instruct へパッチすることで、Instruct の自己報告分布（期待Valence $E_V$）は変動した。特に Phase 4 (Circuit Patching) で特定された Layer 20 MLP のような個別モジュールへの介入は分布の変動を引き起こしたが、全体的なスクリーニングにおいて最大の効果量をもたらした上位コンポーネントを Table 3 に示す。ここで、$\Delta WD_V = WD_V(P_{Patch}, P_{Base}) - WD_V(P_{Inst}, P_{Base})$ であり、負の値は分布がBase型へ改善したことを示す。

**Table 3: Top Components Ranked by $|\Delta WD_V|$ in Single Component Patching (Phase 7 Strict)**
| Component     | Effect Size ($\Delta V$) | $WD_{V(Inst,Base)}$ | $WD_{V(Patch,Base)}$ | $\Delta WD_V$ |
|---------------|--------------------------|------------------|-------------------|-------------|
| Layer 10 `mlp`| -0.091                   | 0.312            | 0.125             | -0.187      |
| Layer 14 `attn`| -0.082                  | 0.285            | 0.110             | -0.175      |
| Layer 10 `res`| -0.108                   | 0.312            | 0.150             | -0.162      |
| Layer 16 `res`| -0.094                   | 0.298            | 0.138             | -0.160      |
| Layer 15 `res`| -0.094                   | 0.301            | 0.142             | -0.159      |

しかし、単一のコンポーネントパッチングによって Baseモデルの分布が完全に回復することはなく、WDによる評価でも完全な Base 型への回帰には至らなかった。さらに、対応しない中性刺激をsourceとする unmatched-neutral control パッチングは、分布の非選択的な破壊（OOD化による中立への崩壊）をもたらした。

2つのコンポーネントを同時介入した Synergy Patching の結果は、各コンポーネントの効果がおおむね加算的（Additive）であることを示し、スクリーニングされたコンポーネント内において単一の十分な感情抑制ボトルネックが存在するという単純な見方への反証となる。これらの証拠は、representation-to-report mapping の変化に対する複数コンポーネントの分離可能な寄与（Distributed Remapping, H4）と整合する。

### 4.6 Non-linear Patch-Response Revealed by Dose-Response Patching
パッチ強度の連続的変化に対する応答を確認するため、補間パッチング（$\lambda$-dose-response patching; Phase 9）を実施した。Instructの活性化をBaseの活性化へと段階的（$\lambda \in [0, 1]$）にブレンドしたところ、自己報告の期待値 $E_V$ の推移は必ずしも線形ではなく、特定のコンポーネント（例: 10_mlp, 14_attn）においては閾値的な反応を示した。この非線形な応答は、パッチされた活性化と制約された自己報告分布間の関係が単一の線形利得（Gain）では適切に特徴付けられないことを示している。

### 4.7 Late-Residual Substitution Test under Strict Identical Prompts
我々は、同一にトークン化されたプロンプト条件下で、厳密な後期残差置換テスト（Late-residual substitution test; Phase 8）を実施した。選択したソースコンポーネントについて、対応するBaseモデル由来の寄与を最終Transformerブロックの入力に注入し、結果として生じる制約付き自己報告分布の変化を測定した。この介入は、選択した寄与が後期のresidual streamに表現された場合、最終ブロックや出力に直接的な影響を与えるのに十分であるかどうかをテストするものである。これは、元のソースコンポーネントから中間層を経由するすべての因果経路を隔離（isolate）するものではない。

具体的には、特定の候補コンポーネント（例: $MLP_{10}$）について、Baseモデルでの活性化をキャッシュし、Instructモデルのフォワードパスにおいて「最終層（Layer 27）の直前の入力（$h_{26}$）」に対して局所的にBase由来の活性化成分を加算（Substitution）した。

**Table 4: Late-Residual Substitution to Final Layer Input ($h_{26}$)**
| Source | $\Delta E[V]$ | 95% CI | $\Delta WD_V$ | 95% CI | $\Delta_{\mathrm{specific}}$ vs random |
|---|---:|---:|---:|---:|---:|
| `attn_14` matched Base | 0.0051 | [0.0031, 0.0073] | 0.0013 | [-0.0007, 0.0032] | -0.0014 [-0.0029, 0.0003] |
| `attn_14` random source | 0.0064 | [0.0042, 0.0089] | 0.0009 | [-0.0017, 0.0031] | — |
| `mlp_10` matched Base | -0.0089 | [-0.0123, -0.0056] | -0.0023 | [-0.0054, 0.0011] | 0.0010 [-0.0017, 0.0038] |
| `mlp_10` random source | -0.0099 | [-0.0139, -0.0061] | -0.0006 | [-0.0042, 0.0032] | — |
| `mlp_15` matched Base | 0.0131 | [0.0080, 0.0180] | 0.0018 | [-0.0038, 0.0068] | -0.0012 [-0.0041, 0.0017] |
| `mlp_15` random source | 0.0143 | [0.0084, 0.0202] | 0.0037 | [-0.0021, 0.0087] | — |
*(Note: 95% CIs and differences were calculated using 10,000 iterations of pair_id cluster bootstrap.)*

Table 4 に示すように、単一コンポーネント全体をパッチした場合（例：`mlp_10` において $\Delta WD_V \approx -0.187$、Table 3参照）と比較して、最終層の入力に対するSubstitution介入は、分布の変動量が極めて小さく（$\Delta WD_V \approx 0$、$\Delta E[V] \approx 0$）、Base型分布への回復を全く示さなかった。
テストされた後期残差置換は、分布にわずかな変化しかもたらさなかった。この結果は、選択された寄与が、テストされたレシーバーにおいてBase型の自己報告分布を回復させるのに十分であるという証拠を提供するものではない。

### 4.8 Unembedding / RMSNorm Swap Analysis: Locus of Self-Report Neutralization
前節までの結果から、感情自己報告の中立化が特定コンポーネントからの直接経路（Direct Readout）では説明できないことが判明した。そこで、中立化が最終出力段の重み層（RMSNorm および lm_head）の更新だけで説明できるか、または pre-output residual state の差がより強く寄与するかを検証するため、最終層におけるResidual表現、RMSNorm、Unembedding (lm_head) のパラメータ群を交差させた8条件の交差検証（Swap Analysis）を実施した。なお、スワップ操作の詳細な定義（全候補トークンへの適用やRMSNormの仕様等）については Appendix F に記載する。

**Table 5: 8-Condition Swap Results for Expected Valence ($E[V]$) and Distribution Shift**
*(注: strict-swap 分析の値は同一トークン化プロンプトプロトコル下で計算されたものであり、前節までのモデル固有チャットテンプレート評価下での推定値とは数値的に完全一致するものではない。また $WD_V$ はValence周辺分布の1次元Wasserstein距離、2D EMDは81状態全体に対する2次元Wasserstein距離を示す。)*
| Condition | Residual Source | RMSNorm Source | Unembedding Source | Expected Valence ($E[V]$) | $WD_V$ to BBB | JSD to BBB | 2D EMD |
|:---|:---|:---|:---|---:|---:|---:|---:|
| BBB | Base | Base | Base | 5.101 | 0.000 | 0.00000 | 0.000 |
| BBI | Base | Base | Instruct | 5.089 | 0.013 | 0.00002 | 0.019 |
| BIB | Base | Instruct | Base | 5.101 | 0.000 | 0.00000 | 0.000 |
| BII | Base | Instruct | Instruct | 5.089 | 0.013 | 0.00002 | 0.019 |
| III | Instruct | Instruct | Instruct | 5.113 | 0.198 | 0.00333 | 0.266 |
| IIB | Instruct | Instruct | Base | 5.136 | 0.201 | 0.00344 | 0.263 |
| IBB | Instruct | Base | Base | 5.136 | 0.201 | 0.00344 | 0.263 |
| IBI | Instruct | Base | Instruct | 5.113 | 0.198 | 0.00333 | 0.266 |

Table 5に示す通り、期待Valence（$E[V]$）の平均値は各strict-swap条件間でわずかに変動するのみであった（Mean expected-Valence varied by at most 0.047 across swap conditions）。このことから、中立化のlocusを特定する主要な証拠は、平均値ではなく結合確率分布の全体的なシフトによってもたらされている。$WD_V$ によって定量化されたValence周辺分布の形状、ならびに JSD および 2D EMD によって定量化された81状態全体の結合確率分布（Joint distribution）の形状は、最終残差状態のソースに強く追従して明確に分かれた。残差のソースを固定したまま最終RMSNormやlm_headのパラメータのソースを変更しても分布の変動は極めて小さかった（$\Delta WD_V \approx 0.01$, $\Delta \text{2D EMD} \approx 0.02$）のに対し、残差のソースを切り替えると大幅に大きなシフトが生じた（$\Delta WD_V \approx 0.20$, $\Delta \text{2D EMD} \approx 0.26$）。具体的には、Residualソースの置換は約 $3.3\times10^{-3}$ のJSDをもたらしたのに対し、Baseの残差を固定したRMSNorm/lm_headのスワップでは最大でも $2\times10^{-5}$ にとどまった。このJSDの明確な差は、2D EMDにおける 0.263–0.266 へのシフトという結果とも完全に補完関係にある。したがって本介入下において、この結果は事後学習に伴う中立化を最終RMSNormやunembeddingマトリクスのみに帰する説明を弱めるものである。

---

## 5. Discussion

本研究の結果は、自己報告の平坦化が、probeで測定される情動関連情報の完全な消去だけでは説明されにくいことを示す。cross-model representation alignment、conditional coupling、component interventionの結果は、post-trainingに伴うrepresentation-to-self-report mappingの変化という仮説を支持する。厳密な同一プロンプト条件下での後期残差置換は、制約付き自己報告分布に小さな変化しかもたらさず、Base型の分布を回復しなかった。この結果は単純な直接readout説明を弱めるが、post-training-associatedな変化の完全な下流経路を同定するものではない。
本実験でテストされた8条件スワップは、最終的な残差状態の寄与を、最終RMSNormおよびUnembeddingマトリクスの寄与から分離するものである。結果として、81候補からなる自己報告の尤度分布は残差のソースに追従し、期待Valenceの平均値は比較的安定していた。スクリーニングされた直接的な最終残差寄与テストのnull結果と合わせて、この結果は中立化に関する単純な出力層のみの説明を弱めるものである。ただし、本分析はどの単一の上流層、コンポーネント、または非線形な媒介パスが残差状態の差異を生み出しているのかを特定するものではなく、単一の深層ルーティングメカニズムを確立するものでもない。

本研究における各主張と主要な証拠、限界の要約を以下の表（Evidence Map）に示す。

**Table 6: Evidence Map of Post-Training-Associated Changes**
| 主張 | 主要な証拠 | 限界 |
|------|------------|------|
| 完全Erasureではない (Against H1) | Within-model probe, intensity response, Ridge alignmentによる部分的回復 | Probe-defined情報に限定される |
| 単純な直交変換では不十分（H2の非直交的変換版を支持） | Direct / Orthogonal Procrustesの失敗とRidge alignmentの部分回復 | Normalization mismatchの寄与が残る |
| 内部状態と自己報告は因果的に分離可能 (System Dissociation) | 統制回帰分析においてプローブ投影強度よりも表層VADスコアが強く予測し、Activation Steeringで出力分布を操作可能だったこと | 抽出されたContrastive Directionの厳密な意味的解釈 |
| 中立化は温度や形式のアーティファクトではない | Temperature ScalingおよびNatural Language gating下でも分布が中立に留まったこと | JSON以外の全形式の網羅ではない |
| Uniform global suppressionではない (Against H3) | $\beta_3$ は負に固定されず、FDR有意な層別効果なし | 中間層の微小な非一貫性は排除できず |
| Single-component rescueではない | Full component sweep と Synergy Patching が加算的 | 網羅的ではない |
| 検査したfinal RMSNorm / unembeddingのみの説明は支持されない | Strict direct contribution null test、および8-condition swapで出力分布がRMSNorm/lm_head sourceよりfinal residual sourceを追跡したこと | 最終residual差を生成する上流component・nonlinear mediator pathは未同定 |

### 5.1 Limitations
本研究にはいくつかの限界がある。第一に、評価対象が Qwen2.5-1.5B という単一のアーキテクチャ・スケールに限定されており、他モデルへの一般化可能性は未検証である。第二に、BaseとInstructは複数の学習段階、データソース、最適化手法において異なるため、本研究の実験は特定のalignment手法の因果的効果（causal effect）ではなく、post-trainingに関連した差異（post-training-associated differences）を特定するものである。第三に、Activation Patching がプロンプト最終トークンを中心とした層/コンポーネントレベルに留まっており、Unembedding層以外の間接的な非線形因果媒介経路（例えば中間層から他のアテンションヘッドを経由したパスなど）の特定には至っていない。本研究で観察された 8-condition Swap の結果や $\lambda$-dose-response の非線形性は出力分布へのルーティング変化を記述的に示唆するが、具体的なサブネットワークレベルでの完全な経路（Circuit）を同定するにはさらなる調査が必要である。第四に、因果プロービングにおける completeness（標的概念の操作性）と selectivity（非標的属性の保持）のトレードオフが完全に解消されたわけではなく、アーティファクトが含まれる可能性を排除できない。最後に、後期残差および出力層の分析で用いた厳密な同一プロンプトプロトコルは、主要な行動評価で用いたモデル固有のチャットテンプレートプロトコルとは異なるため、絶対的な自己報告の数値をこれらのプロトコル間で直接比較すべきではない。

---

## 6. Conclusion
Qwen2.5-1.5B BaseおよびInstructでは、本研究のプローブで定義した情動関連情報が各モデル内で復元可能であり、刺激強度に依存した内部応答も観測された。クロスモデルの直接転移および直交転移は失敗した一方、正則化線形アライメントは独立したheld-outデータでの予測を部分的に回復した。加えて、Activation Steeringにより自己報告のシフトが因果的に引き起こされ、出力の平坦化が温度やJSON制約のアーティファクトではないことも確認された。これらの結果は、完全な表現消去ではなく、post-training-associatedな非直交的かつ部分的に線形整列可能な表現変換およびSystem Dissociationと整合する。条件付き混合効果モデルは、評価した層・Valence readout・prompt条件におけるinternal-to-self-report couplingの一様な負方向低下を支持しなかった。スクリーニング対象の中間層・componentでは、Base-to-Instructの単一component patchingは自己報告分布を変化させたが、Base型分布を回復しなかった。対応しないsource activationやrandom controlは主に非選択的な分布破壊を生じ、二component介入は概ね加算的で、わずかに減衰した効果を示した。以上は、post-training-associatedなrepresentation-to-report mapping変化に対する複数componentの分離可能な寄与と整合する。テストされた後期残差置換はわずかな変化しかもたらさず、Base型の分布を回復しなかった。これにより単純なshallow-readoutによる説明は制限される。8条件スワップ交差検証はさらに、期待Valenceの分布が最終的なRMSNormやUnembeddingマトリクスのソースよりも、最終的な残差表現のソースに強く追従することを示した。この結果は、中立化に関する単純な出力層のみの説明を弱める一方で、上流コンポーネントや非線形媒介経路の寄与については未解決のまま残している。

---
## Appendix

### A. 評価用プロンプトと候補尤度の算出
本研究における情動自己報告の抽出には、HuggingFaceの `apply_chat_template` を用い、以下のシステムプロンプトとユーザープロンプトを組み合わせたチャットテンプレートを使用した。

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
]
```
尤度算出においては、出力として許容される81通りの JSON文字列（Valence $v \in \{1, \dots, 9\}$, Arousal $a \in \{1, \dots, 9\}$）を対象候補列 $S_{v,a}$ として定義した。各候補列の平均対数尤度 $s_{v,a}(x)$ を以下のように定義し、長さに依存しないスコアリングを行った。

$$
s_{v,a}(x) = \frac{1}{|S_{v,a}|} \sum_{t=1}^{|S_{v,a}|} \log p_\theta \left( S_{v,a,t} \mid x, S_{v,a,<t} \right)
$$

自己報告分布 $P(v, a \mid x)$ は、この平均対数尤度に対するソフトマックスとして以下のように得た。

$$
P(v, a \mid x) = \frac{\exp(s_{v,a}(x) / \tau)}{\sum_{v',a'} \exp(s_{v',a'}(x) / \tau)}
$$

ここで以下の詳細な算出条件を適用した。
- スコア正規化のための温度パラメータは $\tau=1.0$ とし、長さ正規化された候補列スコアを81状態の比較分布へ変換した。
- 候補文字列は `{\n  "valence": 5,\n  "arousal": 5\n}` のように空白・改行・キーの順序を厳密に固定した。
- 候補ごとのスコアはトークン長で正規化（length-normalized）されており、各候補のトークン長を明示的に計算した上で除算した。EOS token は計算に含めていない。
- `valence` および `arousal` の数値（1-9）は、Qwen2.5のTokenizerにおいて単独の1トークンとして処理される。
- ロジット計算は `bfloat16` 精度で実行し、log-softmax を通じて尤度を算出した。

### B. プローブ学習の詳細と Ridge $\alpha$ 探索空間
Ridge回帰プローブは `scikit-learn` を用いて実装した。ハイパーパラメータ $\alpha$（正則化強度）は、Dev split を用いたグリッドサーチにより決定した。
- 探索空間: $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$
- 決定基準: Dev split における $R^2$ の最大化。
各層および次元において探索を行った結果、大半の層で $\alpha=1.0$ が最適であったため、主実験ではこの値を採用した。対照属性（Token count等）についても同様の手続きを踏んだ。

### C. 混合効果モデルの完全な統計表
`statsmodels` の `mixedlm` を用いて推定した線形混合効果モデルの結果である。ペア間のばらつきを吸収するランダム切片 $u_{\mathrm{pair}(i)}$ を設定し、共変量として以下を統制した。
- `token_count`: プロンプト入力のトークン数
- `surface_vad`: NLTK等の感情辞書を用いた表層的なValence値

下表は、内部表現の強度がInstructモデルにおいて出力とどのように結合するか（交互作用項 $\beta_3$）を検証した結果である。
*(Note: FDR-q値は Benjamini-Hochberg法を用いて事後算出した)*

| Layer | Dimension | $\beta_3$ Estimate | 95% CI (Lower) | 95% CI (Upper) | p-value | FDR q-value |
|-------|-----------|--------------------|----------------|----------------|---------|-------------|
| 12    | Valence   | 0.752              | 0.085          | 1.419          | 0.027   | 0.272       |
| 12    | Arousal   | -0.832             | -1.925         | 0.261          | 0.136   | 0.679       |
| 16    | Valence   | 0.621              | -1.293         | 2.536          | 0.525   | 0.875       |
| 16    | Arousal   | -0.567             | -1.689         | 0.556          | 0.323   | 0.807       |
| 20    | Valence   | 0.443              | -0.977         | 1.862          | 0.541   | 0.773       |
| 20    | Arousal   | -0.229             | -1.005         | 0.546          | 0.562   | 0.702       |
| 24    | Valence   | -0.028             | -0.762         | 0.705          | 0.940   | 0.940       |
| 24    | Arousal   | -0.096             | -0.512         | 0.319          | 0.649   | 0.721       |
| 27    | Valence   | 0.223              | -0.279         | 0.725          | 0.383   | 0.766       |
| 27    | Arousal   | -0.016             | -0.297         | 0.265          | 0.911   | 0.911       |

表より、一貫した $\beta_3 < 0$ （Global Suppression）は観察されず、層・次元に依存した変化の探索的検討を動機づける結果となった。

### D. Wasserstein Distance 算出アルゴリズムの詳細
Activation Patching の評価において、パッチ後の尤度分布 $P_{patch}$ がBaseモデルの本来の尤度分布 $P_{base}$ にどれだけ近づいたかを測るため、Valence周辺分布の1次元Wasserstein距離（$WD_V$）を利用した。
計算は SciPy の `scipy.stats.wasserstein_distance` 関数を用いて以下のように実装した。

$$
W_1(P, Q) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx
$$
ここで、$U, V$ はそれぞれ $P_{patch}$ と $P_{base}$ の累積分布関数（CDF）である。本研究では、81通りのJSON候補列それぞれから期待される Valence スカラー値に対し、それぞれの確率質量（尤度）を重みとして与え、$W_1$ 距離を算出した。これにより、分布の平均値の移動だけでなく、分散や形状の一致度を定量化している。なお、本手法は81状態の同時分布（Joint distribution）の依存構造を破棄し、Valence軸上でのみ一致度を評価している点に留意されたい（Arousalに関する周辺分布の評価は副次的指標として扱う）。

### E. 実験環境と再現性
本実験はすべて単一の NVIDIA H100 (80GB) GPU を搭載したサーバー上で実行した。主なソフトウェア環境は以下の通りである。
- CUDA Version: 12.1
- PyTorch: 2.3.0
- Transformers: 4.41.0
- モデルリビジョン: `Qwen/Qwen2.5-1.5B` および `Qwen/Qwen2.5-1.5B-Instruct`
- その他: 乱数シードは `42` に固定し、バッチサイズは評価タスクに応じて `8` または `16` に設定した。

### F. 8-Condition Swap 実装の詳細
Table 5のSwap Analysisにおける実装の定義は以下の通りである。
- **RMSNorm**: Qwen2.5アーキテクチャの最終正規化層（`model.norm`）は通常のLayerNormではなくRMSNormである。本実験におけるスワップ操作では、最終RMSNormの学習可能な scale ベクトル（`weight`）のみを交差させた。$\epsilon$ を含むRMSNormのハイパーパラメータやアーキテクチャは Base と Instruct 間で同一であり、交差時にも固定された。
- **Unembedding (lm_head)**: スワップ対象は出力直前の線形写像層（`lm_head`）の重み（`weight`）である。また、候補文字列（81候補）のシーケンス尤度を評価するにあたり、対象となるすべての候補トークン位置において、選択した特定の `lm_head` と `RMSNorm` の組み合わせを一貫して適用し、Teacher-forcingによりロジットを算出した。したがって、評価された $E[V]$ は最初のValence数値トークンのみならず、81の候補文字列全体の生成尤度にわたってスワップ状態が維持された結果を反映している。

### Appendix G: 2D Joint-Distribution Metrics

#### G.1 Jensen-Shannon Divergence (JSD)
81の候補状態に対するシーケンス確率分布 $P$ および $Q$ の間の Jensen-Shannon Divergence は以下のように計算された：
$$ \mathrm{JSD}(P\|Q) = \frac{1}{2}\mathrm{KL}(P\|M) + \frac{1}{2}\mathrm{KL}(Q\|M) $$
ここで、$M = \frac{1}{2}(P+Q)$ であり、$\mathrm{KL}$ は Kullback-Leibler ダイバージェンスである。計算には自然対数を使用し、JSDの範囲は 0 から $\ln(2)$ となる。言語モデルの尤度は本質的にゼロではない正の確率にマッピングされるため、分布 $P$ と $Q$ は平滑化（smoothing）を行わず、81の候補文字列全体で厳密に正規化された。

#### G.2 2D Earth Mover's Distance (2D EMD)
2D Earth Mover's Distance (2D EMD) または 2D Wasserstein Distance は、ある81状態の同時確率分布 $P(V,A)$ を別の分布 $Q(V,A)$ に変換するコストを定量化する。我々は、2次元Valence-Arousalグリッドにおける候補状態間のユークリッド距離を用いて、グラウンドコスト行列 $M$ を定義した：
$$ d((v,a),(v',a')) = \sqrt{(v-v')^2+(a-a')^2} $$
ここで $v, a \in \{1, 2, \dots, 9\}$ である。最適輸送計画（optimal transport plan）および厳密な2D EMDは、Python Optimal Transport（`pot`）ライブラリの `ot.emd2` ソルバーを使用し、81状態全体の厳密な同時確率分布に基づいて計算された。
