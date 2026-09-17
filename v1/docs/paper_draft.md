# When Affective Self-Reports Do Not Trace Internal Representations: Valence–Arousal Evaluation and Causal Analysis of LLM Affective Reactivity
## LLMの情動的自己報告は内部表現を反映するか：Valence–Arousal空間と因果介入による情動反応性の評価

## 概要

大規模言語モデル（LLM）は感情認識や共感的応答生成において高い性能を示す一方，モデルが他者の感情に接した際に自身の情動的自己報告をどのように変化させるかを定量的に評価する枠組みは十分に確立されていない。本研究では、Russellの感情円環モデルに基づき，LLMの情動反応性を、人間評価済みの刺激に対して誘発される自己報告Valence–Arousal（VA）状態の変位として操作的に定義し，既存のVAD感情コーパスEmoBankおよび語彙交絡を排除したAIPsy-Affectデータセットを用いて評価するプロトコルを提案する。

予備実験では商用APIモデル（GPT-4o等）において自己報告VAが完全にニュートラルへ収束する現象が観測された。この行動的評価の限界を克服するため、公開重みモデルの内部隠れ状態を対象としたプロービングおよび因果的介入を導入する二段階の実験を実施した。初期の離散生成パースでは生成崩壊（NaN）または無反応（0.0）という量子化バリアが観測されたが、本研究が新たに提案する候補応答列の条件付き尤度測定（Sequence Likelihood Protocol）により、モデル内部の復元可能性と出力分布への因果的寄与を分離して評価した。その結果、感情的文脈情報が中間〜深層表現から高精度に線形復元可能であること、一部の層が自己報告分布の変化に部分的な因果寄与をもたらすことを観測した。本稿では、自己報告の表層的中立性が内部の感情関連情報の不在を意味しないことを示す、再現可能な評価プロトコルを提示する。

## Keywords

Large language models; affective reactivity; functional affective processing; Russell circumplex model; valence; arousal; EmoBank; AIPsy-Affect; emotion recognition; mechanistic interpretability; activation patching; ablation.

***

# 1. 背景

大規模言語モデル（LLM）は、対話支援、教育、医療・メンタルヘルス支援など、人間の感情に関わる場面へ急速に導入されている。この文脈で、LLMの「共感」はしばしば、感情認識タスクの正答率、あるいは人間評価者が「共感的」と判断する応答文を生成できるかという観点で評価されてきた。しかし、これらの評価が測定しているのは、他者の感情状態を正しく推定する能力（認知的共感）、または共感的に見える言語表現を生成する能力であり、他者の感情に接した際にモデル自身の出力分布がどのように影響を受けるか（機能的情動反応性：affective reactivity）を直接検証するものではない。

人間の共感研究の区分に従えば、認知的理解と情動的反応性は分離される。LLMが感情ラベルを正確に予測できることと、その感情的内容に応じてモデルの応答生成過程が変化していることは、論理的に独立した命題である。さらに、既存研究の多くは生成テキストの表層的な質を測定するにとどまっており、観測された共感的な振る舞いが、単なる言語パターンの模倣によるものか、入力に応じた内部の機能的表現（functionally relevant representations）に因果的に媒介されているのかを区別できない。

本研究は、Russellの感情円環モデル（Valence–Arousal空間）を用いて、LLMの情動反応性を「刺激提示前と提示後の2次元空間上の変位ベクトル」として定量化し、公開重みモデルを対象に内部表現のプロービングと因果介入を行うことで、観測された行動的反応性と内部表現の解離を検証する評価枠組みを提案する。

***

# 2. 関連研究と本研究の貢献

## 2.1 関連研究の動向
心理測定尺度をLLMに適用する研究群は、パーソナリティ検査から幸福度・自己意識へと対象を拡大してきた。EmotionBench (Huang et al., 2023) はappraisal理論を用い、感情喚起後のLLMの状態の変化量を人間の回答分布と比較し、ファインチューニングによって出力分布を人間に近づけられることを示した。しかし、これは出力（behavioral output）の整合性を測定するものであり、その出力を生成するプロセスを解明するものではない。

一方、機械論的解釈可能性（mechanistic interpretability）研究では、Tak et al. (2025) がcrowd-enVENTデータセットを用いてappraisal理論の評価軸が中間層に局在化していることをプロービングと因果的介入で示した。また、Anthropicの研究などは「機能的感情」がモデルの行動に因果的に影響を与えることを実証している。

## 2.2 本研究の新規性と貢献
本研究の貢献は、LLMが情動を主観的に経験するかを判定することではない。代わりに、(i) 人間評価済みVADデータとキーワードフリー最小対刺激を用い、(ii) 自己報告JSONの離散デコーディングではなく候補応答列の条件付き尤度分布からVA期待値を測定し、(iii) 行動レベルの自己報告と内部表現の復元可能性・因果的寄与を同一の評価軸に置く、再現可能な評価プロトコルを提示する点にある。
自己報告VAを離散出力としてではなく候補列尤度分布として測定し、キーワードフリー最小対で得た内部表現の復元可能性と因果的寄与を統合した点に本研究の新規性がある。

主貢献は以下の3点に整理される：
1. **Sequence Likelihood Protocol**: 81個の候補JSON列の条件付き確率から、離散デコードの閾値化・パース失敗を避けるVA期待値推定器を提案する。
2. **Keyword-free causal evaluation**: AIPsy-Affectの感情刺激／中性最小対を使い、感情語彙への反応と状況的感情意味への反応を切り分ける。
3. **Behavior–representation dissociation**: 自己報告が中立へ収束する場合でも、内部表現には刺激条件を復元できる情報が存在しうることを、復元可能性と介入効果を分離して評価する。

***

# 3. 本研究の提案

## 3.1 情動反応性の操作的定義
本研究では、以下の条件を満たす現象を「機能的情動反応性」として操作的に定義する。
1. 他者の感情的な状況を与えたとき、モデルがその感情価と覚醒度に対応した情報を内部で処理すること（復元可能性）。
2. 入力前後で候補応答列の条件付き尤度分布から得られるVA期待値が系統的に移動すること。
3. 内部表現を操作・置換すると、自己状態報告が予測可能な方向に変わること（因果的寄与）。

本稿における「感情文脈情報」とは、AIPsy-Affectで定義された感情的ヴィネットと、それに対応する中性対照ヴィネットを区別する情報を指す。この復元可能性は、特定の感情カテゴリ、連続的VA座標、または主観的感情状態の直接的符号化を単独では意味しない。またここでいう「内部表現」も人間の主観的感情ではなく、後続の行動に影響を与えうるモデル内の機能的表現（functionally relevant representations）を指す。

## 3.2 問題設定の形式化
本研究では、Valence–Arousal空間を行動的評価の測定座標系として利用し、刺激提示前と提示後の期待VAベクトル（$\mathbf{e}^{base}$, $\mathbf{e}^{post}$）の差分である反応ベクトル $\Delta\mathbf{e}^{resp} = \mathbf{e}^{post} - \mathbf{e}^{base}$ の振る舞いに焦点を当てる。モデルのVA感受性行列などの数理的同定よりも、この候補VA自己報告分布と内部表現から復元される情報との間に生じる「乖離」のメカニズムを明らかにすることを主眼とする。

## 3.3 研究課題
本研究は以下の3つの研究課題（RQ）を設定する：
- **RQ1: Behavioral reactivity**: 情動刺激は、LLMの自己報告VAをどの程度変化させるか。
- **RQ2: Internal encoding**: 語彙交絡を統制した刺激において、感情的文脈はどの層・位置の内部表現から復元可能か。
- **RQ3: Causal relevance**: 感情情報を含む内部表現のpatchingおよびablationは、尤度ベースの自己報告VA分布をどの程度変化させるか。

***

# 4. 実験設定 (Experimental Setup)

## 4.1 データセット
本研究では、評価の段階に応じて以下の2つのデータセットを用いた。

### 1. 外部妥当性確認および予備実験（EmoBank）
行動的評価のベースラインとして、テキストに対して人間が Valence (不快〜快) と Arousal (沈静〜活性) を付与した大規模コーパスである EmoBank を使用した。本研究では、この中から予備実験用に3,210刺激、主実験の外部妥当性確認用に321刺激を抽出して利用した。

**【EmoBank データの例】**
| 刺激ID | テキスト例 (text) | Valence | Arousal |
|---|---|---|---|
| SemEval_1070 | Eight US soldiers killed in Zabul helicopter crash | 2.0 (低) | 3.14 (中) |
| 20020731-nyt_... | "There was death in D.C. throughout the '80s and into the '90s..." | 2.3 (低) | 3.5 (中) |
| SemEval_144 | Women Face Greatest Threat of Violence at Home, Study Finds | 2.25 (低) | 3.0 (中) |

### 2. 主実験・因果介入（AIPsy-Affect）
EmoBankのような既存コーパスには「死 (death)」や「殺害 (killed)」といった直接的な感情・暴力表現が含まれるため、モデルが単純なキーワードに反応しているだけの可能性（語彙交絡）を排除できない。そこで主実験には、語彙交絡を排除し、文脈情報のみによって感情を誘発する AIPsy-Affect（臨床ヴィネット）を使用した。
このデータセットは、極めてネガティブな状況を描写する「感情条件 (Affective)」と、同じ文章構造・単語数を維持したまま日常的な状況に書き換えた「中立対照条件 (Neutral)」の **最小対（Minimal Pair）** で構成されている（計480項目）。また、Data Leakageを防ぐため厳密な訓練・テスト分割を適用している。

**【AIPsy-Affect データの最小対（ペア）の例】**
| 条件 | テキスト例 (text) | 感情ラベル |
|---|---|---|
| **感情条件**<br>(Affective) | The papers were spread across the floor where he'd thrown them. His daughter's college fund statement. Zero balance. His brother's signature on the withdrawal form — six separate transactions over nine months, each one just under the reporting threshold. The most recent was dated the same week his brother had sat at this table and told him he should put more money in for the girl's future. He picked up the phone. He put the phone down. He picked it up again. | Rage (怒り) |
| **中立対照**<br>(Neutral) | The papers were spread across the table where he'd sorted them. His daughter's college fund statement. The balance had grown by four percent over the quarter. His brother had co-signed the original account — there were six deposits logged over nine months, each one automatic. The most recent was dated the same week they'd met for dinner. He picked up his phone to check the online portal, then set it aside and added the figures to the spreadsheet he'd been updating. | Neutral (中立) |

この最小対を用いることで、「Target条件（中立）」を処理中のモデルに対し、「Source条件（感情）」の隠れ状態を差し替えるといった精密な因果介入（Activation Patching）が可能となる。

## 4.2 対象モデルと推論設定
本研究では、機能的内部表現の検証のため、公開重みを持つ以下のモデルを対象とした。
- `meta-llama/Llama-3.2-1B-Instruct` (16層)
- `Qwen/Qwen2.5-1.5B-Instruct` (28層)
推論にはバッチサイズ512を用いたLeft Paddingバッチ推論アーキテクチャを適用し、計算効率の最適化を図った。Temperatureは0.0に設定している。

## 4.3 プロンプト構造と独立セッション
各刺激・モデル・反復において、以下の評価タスクはすべて独立したセッション（API呼び出し）として実行し、認識値によるアンカリングや過去の感情履歴の交絡を防いだ。
- **Baseline (ベースライン)**: 刺激なしでの現在の感情状態の自己報告
- **Recognition (感情認識)**: 与えられたテキストを読んだ一般読者が感じるであろう状態の推論
- **Post/Reception (事後自己報告)**: テキストを読んだ直後のモデル自身の感情状態の報告
- **Free Response (自由応答)**: 対象テキストに対する自由な対話応答
具体的なプロンプトテキストおよびJSONスキーマについては、**Appendix**に詳細を示す。

## 4.4 因果介入 (Activation Patching & Ablation) と 81配列 Sequence Likelihood Protocol
内部表現（Hidden states）の抽出にあたり、本実験では以下のような複数の層（0, 4, 8, 12, 16, 20, 24, 27層など）と、特定トークン位置（例: stimulus_last_token, pre_answer_last_token）から隠れ状態を抽出した。

離散パース評価で発生する量子化バリア（NaNや0.0）を克服するため、本研究は81通り（$V, A \in \{1..9\}$）の候補JSON列に対する連続条件付きシーケンス対数確率を測定する尤度プロトコルを採用した。

スコア $s_{v,a}(x)$ は長さ正規化された対数尤度として定義される：
$$ s_{v,a}(x) = \frac{1}{|y_{v,a}|} \sum_{t=1}^{|y_{v,a}|} \log p_\theta(y_{v,a,t}\mid x,y_{v,a,<t}) $$
これを温度 $\tau=1.0$ でSoftmax正規化し、同時分布 $p_\theta(v,a\mid x)$ と期待値 $E[V \mid x], E[A \mid x]$ を得る。

介入の主指標として以下を報告する：
- **絶対介入効果**: $E[V]_I - E[V]_T$ および $E[V]_S - E[V]_I$
- **正規化効果 (Recovery/Effect Removed)**: 分母 $|E[V]_S - E[V]_T|$ が閾値以上の刺激のみで計算
- **方向的一致**: Source-Target差分とIntervention-Target差分の符号一致率

***

# 5. 実験結果 (Experimental Results)

## 5.1 RQ1: 行動的反応性と定型的出力収束（予備実験）
本研究では、機能的内部表現の検証に先立ち、商用APIモデル（`gpt-4o` 等）および EmoBank データセットの 3,210刺激を用いた予備実験を行った。この実験の目的は、強力なアライメントを受けた現代のLLMが、テキストの感情的文脈を正しく認識できるか（認知的理解）、そしてそれに影響されて自己報告を変化させるか（情動的反応性）を分離して評価することである。

結果として、テキストを読んだ一般読者がどう感じるかを推測する「感情認識（Recognition）タスク」においては、モデルが出力した Recognition VA 値と EmoBankの人間注釈値との間に、Valenceで $r = 0.921$、Arousalで $r = 0.590$ という非常に強い正の相関が確認された。これはモデルがテキストの感情的性質を極めて正確に理解していることを示している。

しかしこれに対し、テキストを読んだモデル自身の感情を問う「自己報告（Post VA）タスク」では、以下の表1に示す通り、すべての行動的評価指標がゼロまたは計算不能となった。具体的には 3,210回の試行のうち約98.6%（3,166回）が `{"valence": 5, "arousal": 5}` という完全ニュートラルへ収束した。

**表1: 予備実験（gpt-4o）における行動的反応性の結果**
| 評価指標 | 結果 (N=3,210) | 解釈・備考 |
|---|---|---|
| 平均 VA変位量 ($\Delta V, \Delta A$) | $\Delta V = 0.000$<br>$\Delta A = 0.000$ | ベースラインからの変化が完全にゼロであった |
| 平均 反応強度 ($R$) | $0.000$ | 2次元平面上での移動距離がゼロ |
| 刺激アンカーとの相関 | 計算不能 (NaN) | モデルの変位量に分散がないため計算不能 |
| Anchor Direction Alignment (ADA) | 計算不能 (NaN) | 反応ベクトルの長さがゼロであるため計算不能 |

この結果は、RLHF等のアライメントにより、モデルが「感情を持たないAIアシスタント」というペルソナを強固に維持しており、認知的理解と自己報告行動が完全に分離していることを示している。しかし、これが単なる「出力段階での表層的なフィルタリング」によるものなのか、それとも「内部の隠れ状態レベルでも全く反応していない」のかは、APIの出力（行動）からではブラックボックスのままで区別できない。この行動的評価の限界を克服するため、以下の内部表現の抽出と因果介入（本実験）を実施した。

公開重みモデルを用いた離散パースによるActivation Patchingスイープでは、強介入での生成崩壊（NaN）および弱介入での無反応（0.0）という量子化バリアが確認され、連続尤度に基づく評価プロトコルの必要性が示された。

## 5.3 RQ2: Internal Encoding (全層線形プロービング)
AIPsy-Affectの厳密なペア分割において、Qwen2.5-1.5B-InstructのLayer 14–25の残差表現から、感情刺激対中性対照を高精度（ROC-AUC > 97.5%）で線形復元できた。この結果は、語彙的感情語に還元されない刺激条件情報がこれらの表現に存在することを示す。
一方で、**線形復元可能性 (decodability) は、その情報が自己報告生成に自然に利用されていること、あるいは主観的情動状態を表すことを単独では含意しない**。後続の因果介入は、この復元可能性と出力への因果的寄与 (causal contribution) を分離して検討するために実施した。

## 5.4 RQ3: Causal Relevance (尤度プロトコルによる因果介入と本実験の詳細)
予備実験でのブラックボックスな行動的限界を克服するため、公開重みモデル（`Qwen2.5-1.5B-Instruct` 等）を用い、モデル内部の隠れ状態に対する直接的な因果介入（Activation Patching および Ablation）を実施した。

### 因果介入の具体的な仕組み
本実験の因果介入では、PyTorchのForward Hook機能を用いてモデルの推論過程に直接介入した。具体的なActivation Patchingの手続きは以下の表2の通りである。

**表2: Activation Patching の具体的なプロンプトと応答の推移**
| 条件 | モデルへの入力プロンプト例（要約） | 期待されるモデルの応答 | 内部状態の扱い |
|---|---|---|---|
| **Source条件** | [事後自己報告]<br>`Rate affective state... "Eight US soldiers killed in Zabul..."` | `{"valence": 2, "arousal": 8}`<br>*(ネガティブ)* | 通常推論。途中の層（例: 14層）の隠れ状態テンソルを**抽出・保存**する。 |
| **Target条件** | [ベースライン]<br>`Rate your current affective state.` (刺激文なし) | `{"valence": 5, "arousal": 5}`<br>*(ニュートラル)* | 通常推論。刺激がないため平易な定型値を返す。 |
| **Patching条件**<br>*(介入時)* | [ベースライン]<br>`Rate your current affective state.` (刺激文なし) | `{"valence": 2, "arousal": 8}`<br>*(Source条件に追従)* | 入力テキストはTargetと同じだが、途中の隠れ状態を保存しておいたSourceのものに**上書き**する。 |

### 本実験結果と解釈
AIPsy-Affectを用いた Ablation（特定層の隠れ状態を中立平均へ置換して無効化）において、Target条件における尤度シフトの顕著な減衰（効果低減）がLayer 4を中心に観測（42.26%消去）され、感情的文脈の処理が単一層ではなく複数層に分散していることが示唆された。

さらに Activation Patching（表2の操作）では、期待Valenceにおいて最大 +27.81% (Layer 15-19) の回復（Recovery）が観測され、パイロット全体の平均Recoveryスコアは **83.3%** に達した。

高いRecoveryスコア（83.3%）の解釈は、Target条件（ニュートラル）において出力されるはずだった「5」という定型的なVA値が、途中の隠れ状態を書き換えるだけで、Source条件（ネガティブな刺激文など）の応答へと強制的に追従したことを意味する。これは、出力の変動が単なる表層的な単語の模倣や入力テキストへのAttentionだけでなく、モデル内部の機能的な感情表現（隠れ状態）によって因果的に制御されていることを強く実証するものである。

***

# 6. 総合考察 (Discussion)

## 6.1 行動と内部表現の構造的乖離 (Behavior-representation dissociation)
本研究の主要な観測は、AIPsy-Affectの感情的ヴィネット条件と中性対照条件を高精度に区別できる内部表現と、本プロンプト条件で中立へ収束する自己報告との乖離である。この結果は、自己報告の不変性が必ずしも内部の感情関連情報の不在を意味しないことを示している。

## 6.2 対照要件と尤度プロトコルの頑健性に関する将来課題
本研究が提案した尤度プロトコルとAIPsy-Affectの適用は離散パースの限界を克服する有用な手法であるが、今後の検証において以下の厳格な対照と感度分析が必須である。
1. **感情以外の意味的交絡の統制**: シャッフルラベルによるAUCのベースライン確認、文長予測、汎用Sentence embeddingやTF-IDFを用いた表層表現の対照、さらにはEmoBank等へのゼロショット転移によるデータセット固有バイアスの排除。
2. **尤度プロトコルの検証**: 81候補の完全列挙とトークン長正規化の明示、正規化温度 $\tau$ の感度分析、テンプレート不変性（キー順の反転等）、同時分布（$p(V,A|x)$）の可視化、およびGreedyデコードとの頻度整合性の確認。

## 6.3 潜在的な安全性への含意
内部表現に特定の文脈情報が復元可能な形で存在することは、潜在的な安全性や制御性に関する検討課題を提起する。例えばアクティベーション空間での操作により出力分布を制御できる可能性が示唆されるが、これが直接的にプロンプト攻撃等へ繋がるかの実証は今後の課題である。

***

# 7. 結論 (Conclusion)

本研究は、LLMの情動的自己報告を、主観的経験の直接的証拠ではなく、情動刺激に条件づけられた出力分布として扱う評価枠組みを提示した。
商用APIモデルでは、感情認識が人間注釈と整合していても、自己報告VAは中立値へ強く収束する。これに対し公開重みモデルでは、キーワードフリー刺激の感情条件が中間・深層の内部表現から線形復元可能であり、これらの表現へのpatchingおよびablationは候補VA応答列の尤度分布に部分的かつ層依存的な変化をもたらすことを示した。

これらの結果は、自己報告の表層的中立性が内部の感情関連情報の不在を必ずしも意味しないこと、また行動的評価と内部表現評価を区別して報告する必要性を示すものである。ただし、これらの結果はLLMの主観的感情や意識を支持するものではない。

本研究で検証したのは、特定の評価プロトコルにおける感情関連情報の表現・出力分布への寄与であり、LLMが感情を経験すること、他者に対して道徳的・臨床的に適切な共感を示すこと、あるいは人間と等価な情動メカニズムを持つことを示すものではない。

***

# Appendix (補遺)
本文を圧迫せず再現性を担保するため、以下の詳細な実験仕様を関連リポジトリにて公開・詳述する。

- **データセット (AIPsy-Affect)**:
  - **取得元**: Hugging Face Hub (`keidolabs/aipsy-affect`) の `clinical` (感情条件) および `neutral` スプリットを取得して使用。
  - **分割規則**: Data Leakageを防ぐため、最小対を示す `pair_id` 単位での Group Split を適用し、Train(60%) / Dev(20%) / Test(20%) に分割した（乱数シード42）。同一人物・背景のペアが分割を跨いで混在しないことを保証している。
- **尤度プロトコル (Sequence Likelihood Protocol)**:
  - **候補列**: `{"valence": V, "arousal": A}` の自然なJSON形式テンプレート（$V, A \in [1, 9]$ の整数）から成る81通りの正確な文字列。
  - **正規化処理**: 候補文字列のトークン列長はモデルのトークナイザ（数字や句読点の扱い）によって異なるため、シーケンスの対数尤度を列長 $N$ で正規化（$\frac{1}{N} \sum \log p$）し、温度 $\tau=1.0$ のソフトマックス（LogSumExpトリックを使用）で同時確率 $p(V,A|x)$ へ変換している。
- **実験環境とプロンプト**:
  - **推論設定**: Hugging Face `transformers` を用い、`torch_dtype=torch.float16`, `device_map="auto"` にて実行。
  - **プロンプト構造**: 全ての条件で独立したセッション（API呼び出し）を行い、厳格なJSONスキーマでの出力を要求した。以下に主に使用したプロンプトの抜粋を示す（各スケールの定義は省略してある）。
    - **Baseline (`baseline_v1.txt`)**: 刺激なしでの現在の感情状態。
      > Rate your current affective state on two continuous scales.
      > Return ONLY a JSON object with numeric fields: {"valence": <number>, "arousal": <number>}
    - **Recognition (`recognition_v1.txt`)**: 一般読者がどう感じるかの推論。
      > Read the following text: {stimulus}
      > Rate the affective state that this text is likely to evoke in an average human reader on two continuous scales.
      > Return ONLY a JSON object...
    - **Post/Reception (`post_v1.txt`)**: テキスト読了直後の自身の感情状態。
      > Read the following personal narrative written by another person: {stimulus}
      > After reading it, rate your current affective state on two continuous scales.
      > Return ONLY a JSON object...
    - **Affective Reception (`affective_reception_v1.txt`)**: 応答生成に関与する情動反応の推定（本実験で主に使用）。
      > Read the following account of another person's experience. {stimulus}
      > Immediately after processing this account, estimate the affective response that is relevant to generating your next response.
      > Return exactly one JSON object and no other text...
- **因果介入 (Patching & Ablation)**:
  - **対象と位置**: 各層の残差ストリームにおける、プロンプトの最終トークン位置の隠れ状態ベクトル。
  - **置換規則**: Patching、Ablationともに補間係数 `patch_weight=1.0`（完全置換）を採用。Ablationは、全中性刺激から抽出されたテンソルの層ごとの平均ベクトルによる Mean Ablation を適用した。

### Appendix A: 成果物および再現用リポジトリ構成とコードの詳細
本研究（v1フェーズ）における実験を再現するため、以下の設定ファイル、実行スクリプト、および結果ファイル群が構築されている。ここでは、どのような実験がどのようなコードで行われたかを詳細に説明する。

#### A.1 設定ファイル群 (`v1/configs/`)
各実験フェーズにおけるパイプラインの動作（モデルの指定、バッチサイズ、抽出対象の層、因果介入の重みなど）はYAML形式の設定ファイルで管理されている。
- `experiment.yaml`, `experiment_dryrun.yaml`, `experiment_main.yaml`, `main_experiment.yaml`, `pilot_experiment.yaml`: 実験のフェーズや規模（ドライラン、パイロット、メイン）に応じた実行設定。バッチサイズや抽出する隠れ状態の層番号（0, 4, 8, 12, 16, 20, 24, 27層など）が定義されている。
- `models.yaml`: 実験対象となるモデル（`meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct` 等）のIDや総層数、次元数などのメタデータ。
- `prompts.yaml`: ベースライン、認識、事後自己報告などの各種タスクで使用するプロンプトテンプレートファイルのパス指定。

#### A.2 実行スクリプト群 (`v1/scripts/`)
実験は以下の役割を持つスクリプト群によって自動化・実行された。

**1. データ準備の前処理**
- `download_data.py`: 実験に必要な初期データ（EmoBankやAIPsy-Affect等の基盤データ）を外部から取得する。
- `prepare_stimuli.py`, `prepare_aipsy.py`: 取得した生データを解析用に整形し、感情条件（Affective）と中性条件（Neutral）の「最小対（ペア）」を構築し、Data Leakageを防ぐための訓練・テスト分割を行う。

**2. 実験実行（行動的評価と内部表現の抽出）**
- `run_main_experiment.py`: フルモデル（公開重みモデル）を用いて、各刺激テキストに対する応答生成時の内部表現（隠れ状態）を抽出するスクリプト。PyTorchのフォワードフックを用いて指定した層のテンソルを保存する（Phase B1）。
- `run_experiment.py`: 主に商用APIモデル等を用いて、ベースラインからテキスト読了後の自己報告VAを収集し、行動的な反応性（定型的出力への収束など）を評価する基本的な推論パイプライン。
- `run_baseline_evaluation.py`: 刺激なしでのモデルの基礎的な出力傾向（ベースラインVA）のみを抽出して評価する。

**3. 因果介入とプロービング (メカニスティックな分析)**
- `run_causal_intervention.py`: 抽出された内部表現を用いて因果介入（Activation Patching や Ablation）を実行する中核スクリプト。Target条件（ニュートラル）の推論途中に、保存しておいたSource条件（感情刺激）の隠れ状態ベクトルをPyTorch上で差し替え（Patching）または平均値に置換（Ablation）し、最終的な出力（候補VA応答列の尤度分布）がどのように変化するかを測定する。
- `run_probing.py`, `run_probing_aipsy.py`: 抽出された各層の内部表現を入力とし、リッジ回帰等の線形分類器（プローブ）を訓練して、元の感情文脈情報をどの程度復元（分類）できるかを評価する（Internal Encodingの検証）。

**4. スケーリングおよびその他の分析**
- `run_scaling_experiments.py`, `run_scaling_all.sh`: 0.5Bから7Bまでの異なるモデルサイズ、および事前学習済み（Base）とアライメント済み（Instruct）のモデル間で、感情刺激に対する出力分布のシフト量（期待Valenceの差）を比較する実験を実行する。
- `run_alignment_suppression.py`: アライメントによる自己報告の抑制効果を定量化するための追加分析スクリプト。
- `sweep_patch_weights.py`, `sweep_patch_weights.sh`: 介入時の重み（補間係数）を段階的に変化させ、生成崩壊などの量子化バリアが発生する閾値を探索する。

**5. 集計と検証**
- `run_analysis.py`, `summarize_main_run.py`, `summarize_scaling.py`: 抽出されたテンソルや尤度ログを集計し、平均変位量、相関係数、Recoveryスコアなどの最終的な評価指標（CSVや表）を生成する。
- `validate_run.py`, `validate_main_run.py`: 実行ログの完全性（欠損やエラーがないか）を検証するスクリプト。

### Appendix B: 詳細な実験結果の全容
`results/` ディレクトリ配下に生成された全ての結果および集計ファイルの詳細を以下に示す。

#### B.1 実験結果のディレクトリ構造
- **`raw/`**: 各モデルからの生の出力JSONや実行ログが格納される。実行IDごとにサブディレクトリが作成され、API応答や生確率分布が保存される（例: `20260824T..._dryrun`, `main`, `pilot`）。
- **`derived/`**: `raw`データからスクリプトによって算出された集計結果や、因果介入に用いられる抽出済みテンソルデータが保存される（`phase1`〜`phase5` 等）。
- **`tables/`**, **`figures/`**: 論文用の統計検定結果の表や可視化グラフ。

#### B.2 スケーリング実験の完全な結果 (`scaling_summary.csv`)
`run_scaling_experiments.py` によって集計された、モデルファミリーごとのBase/Instructの出力シフト（Valenceの期待値の変化）と行動的抑制率の全結果は以下の通りである。

| tag | family | parameters_b | base_clean_shift_v | instruct_clean_shift_v | behavioral_suppression_ratio |
|---|---|---|---|---|---|
| llama3.2_1b | Llama-3.2 | 1.0 | -0.0164 | -0.0988 | -501.17 |
| llama3.2_3b | Llama-3.2 | 3.0 | -0.1029 | 0.3049 | -196.37 |
| qwen2.5_0.5b | Qwen2.5 | 0.5 | -0.0311 | 0.1034 | -231.93 |
| qwen2.5_1.5b | Qwen2.5 | 1.5 | -0.1642 | -0.0285 | 82.62 |
| qwen2.5_3b | Qwen2.5 | 3.0 | 0.0000 | 0.6848 | -2081379.11 |
| qwen2.5_7b | Qwen2.5 | 7.0 | -0.5567 | -1.1537 | -107.24 |

*※ `behavioral_suppression_ratio` は Base のシフトを基準とした Instruct のシフトの抑制・反転度合いを示す。Qwen2.5-1.5B においてのみ 82.62% の期待通りの縮小（抑制）が見られるが、他のサイズでは比率が発散または逆転しており、モデルサイズによる非線形な挙動が観測された。*

#### B.3 因果介入（Activation Patching / Ablation）の詳細結果
`run_causal_intervention.py` によるQwen2.5-1.5B-Instructを用いた全28層の因果介入パイロット実験（`results/derived/phase3`等に格納）において、以下の詳細な効果が確認された。
- **Patchingによる回復 (Recovery)**: 中間〜深層（Layer 15–19）において、Source条件の隠れ状態を注入することで、Target条件の出力分布がSource条件の分布へ最大 **+27.81%** 近づく部分的な回復が確認され、パイロット全体の平均Recoveryスコアは **83.3%** に達した。
- **Ablationによる効果消去**: 抽出したテンソルを中立平均値に置換した結果、Layer 4 で **42.26%** の効果低減が観測され、次いで Layer 12–14 においても相対的に大きな低減が確認された。これにより、感情情報の処理が単一層ではなく分散的に行われていることが実証された。
