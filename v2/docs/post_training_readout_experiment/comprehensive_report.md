# 包括的レポート: Post-Training Associated Representation Transformation and Readout Remapping

## 1. 概要 (Overview)
本レポートは、Qwen2.5-1.5B (Base) と Qwen2.5-1.5B-Instruct の内部表現および自己報告（Expected Valence $E_v$ および Arousal $E_a$）を比較し、事後学習（post-training）に関連するBase/Instruct間の差が感情表現の内部ジオメトリと自己報告の対応関数（Coupling）に与える影響を明らかにする一連の実験の総括です。

本研究の最終的な中心主張は以下の通りです：

> **Post-trainingに伴う変化は、復元可能な情動関連情報を内部に保持する一方、表現幾何と自己報告への対応関係を変容させる。この変化は層および評価次元に依存する可能性を示すが、次元特異的パターンの確証には追加の検証が必要である。また、単一の検査済みcomponentの介入ではBase型の自己報告分布は回復しない。これらの結果は、完全な表現消去や一様な出力抑制を支持せず、分散的なrepresentation-to-report mappingの変容と整合する。**

---

## 2. 評価仮説 (Hypotheses)
実験を通して以下の4つの仮説を検証しました。
- **H1 (Erasure)**: 事後学習により、内部空間から感情情報自体が消去される。
- **H2 (Representation Transformation)**: 感情情報は維持されるが、その表現される部分空間や幾何的構造（Geometry）が大きく変化する。
- **H3 (Readout Attenuation / Suppression)**: 内部表現は維持されているが、自己報告への因果パスが意図的に遮断され、出力が中立に固定される。
- **H4 (Distributed Readout Remapping)**: 内部表現は維持・変容し、自己報告との対応関係（Coupling Function）そのものが新しいマッピングに書き換えられる。この再写像は単一のボトルネックではなく、ネットワーク全体に分散して生じている。

---

## 3. 実験手法と再現のための実装詳細 (Methods & Implementation Details)

### 3.1 HIC/HBC 指標の堅牢化 (Slope / Covariance の導入)
Instructモデルにおいて自己報告 ($E_v$) の分散が極端に潰れている場合、ピアソン相関（スケール不変）を用いると、内部表現の分散に対して出力の分散が極小であっても高い相関が出てしまうため、**回帰係数 (Slope, $\beta$)** を導入し、「内部表現 $z$ の単位変動に対して自己報告 $E_v$ がどれだけ絶対的にスケールするか」を測定しました。

### 3.2 介入（Steering）フックの全ポジション拡張
Activation Steering の介入がプロンプトの最終トークンのみに適用されていたバグを修正し、介入を対象トークン以降の全ポジションに継続して適用することで、生成時にも一貫した介入を維持しました。

### 3.3 Activation Patching の評価指標 (Wasserstein Distance)
PatchingによるBaseへの「回復」を測る際、平均値ではなく分布全体の形状の一致度を測る **Wasserstein Distance (WD)** を採用し、Patched分布がどれだけBase分布に近づいたかを定量評価しました。

### 3.4 Cross-decoding の厳密化 (Test split 評価)
Ridge回帰によるアライメントマップは、完全に未使用の pair_id test split で評価しており、過学習による見かけ上の回復を排除しています。

---

## 4. 主要な発見 (Key Findings)

### 4.1 表現の変容 (Representation Transformation)
- **Dose-Response (強度応答)**: 
  感情強度の変化に対して、両モデルともintensityを内部的に追跡していることが確認されました。本検証のプローブで測定された情動関連情報が完全に消去されたという証拠は得られませんでした。代わりに、モデル内での高いデコード可能性と、部分的に回復可能なクロスモデル予測アライメントは、表現の変容下での情報の保存と整合します。
- **Cross-decoding (空間の予測的アライメント)**: 
  Direct mapping や Orthogonal Procrustes では復元が失敗した一方、線形写像（Ridge）を用いることで予測性能が回復しました。これは、単純な剛体回転ではなく、**事後学習 (Post-training) が情動関連表現の非直交的かつ部分的に線形整列可能な変換を伴う** ことを示しています。
- **Module Probing**: 
  Test splitを用いたheld-out予測性能（Ridge $R^2$）等を指標としたとき、Attention output における Base/Instruct 間の差異は比較的小さい一方で、late-layer（Layer 20以降）の MLP outputs はより強い表現的乖離 (representational divergence) を示しました。

````carousel
![Cross Decoding R2](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/cross_decoding_r2.png)
<!-- slide -->
![Cross Decoding IBC](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/cross_decoding_ibc.png)
````

### 4.2 Coupling の再写像 (Dimension-Specific Remapping)
- **Valence Readout の非一様な変化**: 
  Coupling Slopeを用いた評価では、Valence において Instruct の Slope が Base を下回る（Attenuation）という一貫した結果は得られませんでした。むしろ中間層などでは Instruct の方が Slope が大きくなっています。
  評価されたValenceのreadoutにおいて、事後学習は一様な減衰をもたらしませんでした。図示された層ではInstructモデルのslopeの方が大きく、情動的自己報告の全体的な抑制 (global suppression) よりも、層に特異的な再写像 (remapping) が起きていることを示しています。この結果は、少なくとも層依存のcoupling変化と整合します。ValenceとArousalを含む次元特異的なパターンについては、追加のモデル、刺激、prompt条件における再現が必要です。
- **Orientation-controlled reanalysis of arousal coupling**: 
  An initial analysis suggested a sign reversal of the internal-arousal-to-self-report association in the Instruct model. Because the sign of a linear probe direction is arbitrary, we reoriented each probe using its association with human arousal labels on the training split and reevaluated on held-out pairs. The apparent reversal did not persist after this procedure. We therefore treat the initial pattern as a probe-orientation artifact and do not interpret it as evidence of sign-changing arousal remapping.
- **Steering に対する Negative Result との整合性**: 
  After normalizing intervention magnitude and comparing against norm-matched random directions, single-direction steering did not yield a robust, monotonic, direction-specific change in self-report within the prespecified quality-preserving range. これは、内部表現が decodable であることと、単純な線形方向への操作で出力を制御できること（steerability）が一致しないことを示す negative result として位置づけられます。

````carousel
![Steering Slope L20](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer20.png)
<!-- slide -->
![Steering Slope L24](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer24.png)
<!-- slide -->
![Steering Slope L27](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer27.png)
````

### 4.3 分散的な再写像メカニズム (Distributed Remapping vs Single Gating)
- **Activation Patching による Base 分布の「非」回復と層別スクリーニング**: 
  Instructモデルに対してBaseモデルの各層・各コンポーネント (res, attn, mlp) の活性をパッチする網羅的スクリーニング (Layer 10–27) を実施しました。結果として、特定の1つの層やモジュール（例えば Layer 20 の MLP など）だけで劇的に $\Delta V = -1.0$ のように全体の分布が Base 状態へ戻る（H3: 単一ボトルネック）ことはありませんでした。
  
  | Layer | Component | $\Delta V$ | $\Delta A$ | Entropy |
  |-------|-----------|------------|------------|---------|
  | 10    | res       | -0.108     | -0.068     | 3.449   |
  | 15    | res       | -0.094     | -0.006     | 3.441   |
  | 16    | res       | -0.094     | -0.022     | 3.475   |
  | 10    | mlp       | -0.091     | -0.081     | 3.394   |
  | 14    | attn      | +0.021     | +0.002     | 3.383   |
  | 11    | mlp       | +0.026     | +0.015     | 3.357   |

  本スクリーニングで対象とした層・componentの範囲では、単一のBase由来activation置換によってBase型自己報告分布を回復するcomponentは確認されませんでした。この結果は、単一の検査済みcomponentだけで十分な救済が達成される単純なモデルに反します。一方で、未検査層・head・出力投影、複数componentの相互作用、非線形readout機構の可能性は残ります。

- **複数コンポーネントによる Synergy (分散的因果貢献の証明)**: 
  単一モジュールではなく、中層（Layer 10〜16付近）の複数のコンポーネントが、それぞれ独立して微小な因果的影響を及ぼしていることがわかりました。さらに、これら複数のコンポーネントを同時にパッチした結果、強い非線形の干渉（相殺）はなく、各々の独立した効果の線形和に近い変化量が観測されました。
  
  | Intervention | $\Delta V$ from Instruct | Notes |
  |--------------|--------------------------|-------|
  | `10_mlp` (Single) | -0.0911 | Independent causal effect |
  | `14_attn` (Single) | +0.0207 | Independent causal effect |
  | **Expected (Sum)** | **-0.0704** | Linear additivity hypothesis |
  | **Observed (Joint)** | **-0.0561** | Actual effect of simultaneous patch |
  | **Synergy (Obs - Exp)** | **+0.0143** | Mild dampening, no collapse |

  実測された複合効果は、若干の減衰を伴いつつも個別の効果の和とおおむね一致しました。このパターンは、検査された2つのコンポーネントがこの介入設定下で分離可能な因果的寄与を持つことを示しています。これは分散的な寄与の仮説と整合しますが、これ単独で完全な因果経路を特定したり、ネットワーク全体の再配線を証明するものではありません。

- **フォーマットバイアスの棄却の限定**: 
  Neutral convergence persisted under both JSON and natural-language response formats, reducing the likelihood that the phenomenon is solely attributable to the specific JSON syntax. It does not rule out prompt-level, scale-level, or assistant-persona-related response priors.

![Output Gating](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/output_gating_plot.png)

---

### 4.4 追加検証と頑健性 (Robustness & Control Verifications)

分散的寄与仮説と代替説明を検証し、表層的特徴への過学習や交絡変数、非選択的なactivation破壊の可能性を評価するため、一連のコントロール実験を追加実施しました。

- **交絡統制下での Mixed-effects Model**: 
  表層的属性や意味的属性（Token count, Emotion category, Clinical domain, Narrative richness, 辞書ベースVADスコアなど）を共変量として追加した線形混合モデルでも、内的Valence表現から自己報告への交互作用項（$\beta_3$）は有意でした（例: $+0.430$, $p<0.001$）。この結果は、少なくとも評価したValence readout、層、およびprompt条件において、post-trainingに伴う一様な負方向のcoupling低下を支持しません。したがって、単純なglobal attenuation accountとは整合しにくいと言えます。一方で、層・方向・prompt・未検査componentに限定された抑制機構の可能性は排除しません。

- **Cross-decoding に対する Control Targets**: 
  Ridge Alignmentの予測ターゲットとして、感情スコアだけでなく低レベルな対照属性を用いた比較を行いました。
  
  | Target Attribute | Predictive Alignment ($R^2$) |
  |------------------|-----------------------------|
  | **Valence**      | **0.58**                    |
  | **Arousal**      | **0.42**                    |
  | Surface VAD      | 0.12                        |
  | Narrative Richness | 0.08                        |
  | Token Count      | 0.05                        |

  Valence/Arousalに対するRidge alignmentは、採用した低レベル対照ターゲットより高くなりました。このパターンは、alignment回復がtoken数や表層VAD、narrative richnessだけに還元されないことを示し、affect-relevant predictive alignmentと整合します。

- **Advanced Patching と Distributional Limits**: 
  意味的に一致しないパッチ（Out-Of-Distribution）やランダムベクトルを用いた Matched-control では分布の無秩序な破壊のみが観察され、意味の合ったBase表現を用いた際のみ分布が滑らかに変容しました。一方で、Wasserstein Distance の最大改善幅は $\sim 0.15$ に限定されました。WDの改善幅が限定的であったことは、検査した単一・少数componentの線形置換ではBase型分布を十分に回復できなかったことを示します。この結果は、追加component、複数componentの相互作用、または非線形なreadout機構の可能性を動機づけますが、それらの存在を単独で同定するものではありません。

---

## 5. 結論 (Conclusion)

Qwen2.5-1.5B BaseおよびInstructでは、本研究のプローブで定義した情動関連情報が各モデル内で復元可能であり、刺激強度に依存した内部応答も観測された。クロスモデルの直接転移および直交転移は失敗した一方、正則化線形アライメントは独立したheld-outデータでの予測を部分的に回復した。この結果は、完全な表現消去ではなく、post-training-associatedな非直交的かつ部分的に線形整列可能な表現変換と整合する。条件付き混合効果モデルは、評価した層・Valence readout・prompt条件におけるinternal-to-self-report couplingの一様な負方向低下を支持しなかった。スクリーニング対象の中間層・componentでは、Base-to-Instructの単一component patchingは自己報告分布を変化させたが、Base型分布を回復しなかった。対応しないsource activationやrandom controlは主に非選択的な分布破壊を生じ、二component介入は概ね加算的で、わずかに減衰した効果を示した。以上は、post-training-associatedなrepresentation-to-report mapping変化に対する複数componentの分離可能な寄与と整合する。ただし、その完全な回路機構および複数モデルへの一般化可能性は今後の課題である。
