"""
scratch/fix_paper_md.py
Repairs encoding errors and cleans up duplicated sections in v3/docs/paper.md.
"""

import re

file_path = "v3/docs/paper.md"

with open(file_path, "rb") as f:
    raw_data = f.read()

# Decode with ignore to strip invalid bytes
text = raw_data.decode("utf-8", errors="ignore")

# Let's see what's in text between line 315 and 365
lines = text.splitlines(keepends=True)
print(f"Total lines: {len(lines)}")

# Find where the corruption is
for i, line in enumerate(lines):
    if "Section 3で言及した予備的な全層スイープ実装" in line:
        print(f"Corrupted line {i+1}: {line[:100]}...")

# Clean text:
# Between line 319 ("*プローブ推定値の整合性に関する注記*:") and "### 6.2 Robustness to Likelihood Scoring"
# There should be:
# - the note on probe consistency
# - the Section 6.1 hypotheses (Stage 1 and Stage 2 focused evaluation)
# - the summary boxes

correct_section_6_1_text = """*プローブ推定値の整合性に関する注記*:
本研究の主解析では、全28層・全コンポーネントで統一的に算出した $R^2_{\\mathrm{MLP},15} = 0.5610$、$R^2_{\\mathrm{ATTN},18} = 0.5495$、$R^2_{\\mathrm{RESID},14} = 0.5016$ を正本として採用している。なお、Section 3で言及した予備的な全層スイープ実装（初期の尤度ロバストネス評価パイプラインと同時に抽出されたプローブ推定: 0.546）と比較しても、定性的な層別プロファイルおよび効果量の範囲は緊密に整合している。

さらに、本研究では以下の対立仮説を独立実験により二段階で検証・評価した：

- **Stage 1: Exploratory Generation-time Screen (15-pair Sweep)**: 初期の15-pair全層探索スクリーニングでは、生成プレフィックス直後でのパッチングにより後段層（L24 MLP: 5.02%, L20 ATTN: 4.23%）で微小な変位上昇傾向が観測されたものの、単一層の回復率は全体として小さく疎（有効13ペアの中央値 0.00%）であった。この初期スクリーニングは推定値のばらつきを含んでいたため、代表層における全数コホート評価を動機づけた。
- **Stage 2: Focused Full-Cohort Evaluation on Representative Layers**: 初期スクリーニングで得られた候補層に加え、probe peakおよびcomponent-specific representative layersを含む代表6層（L10, 14, 15, 18, 20, 24）について、Held-out test全39組の完全一致ペアを評価対象とした（`v3/results/focused_causal_sweep_39pairs.csv`）。分母セーフガード適用後、Prompt時32ペア、Generation時39ペアが有効であった。その結果、プローブ最高層 Layer 15 MLP（$R^2=0.5610$）の回復率は有効32 pairs平均でPrompt時 **0.06%**（中央値 0.07%）、Generation時でも **1.39%**（中央値 0.50%）に留まり、中盤層線形表現の局所十分性の低さが確認された。一方、生成時においては後段層の **Residual Stream（L18: 41.48%, L20: 50.22%, L24: 53.48%）および後段MLP（L24: 15.23%）** に実質的な因果回復が出現した（substantial recovery emerged）。対照的にAttentionは後段層でも回復率が負値〜微小（L18: -7.74%, L20: 1.01%, L24: -0.07%）であり、単一層射影出力からの限定的回復が確認された。
- **Probe-Aligned Necessity Sweep & N=100 Focused Test**: プローブ方向の直交射影消去による中和比率は全層で -3.27%〜+0.61% であり、系統的な中和傾向は認められず、直交ランダム方向に対する特異性検定（BH-FDR補正後、全84条件）で有意な層は皆無（最小 $q = 0.857$）であった。さらに、重要代表5層（L10, 15, 18, 20, 24）における **N=100 ランダム直交方向高解像度追試**（`v3/results/focused_necessity_sweep_n100.csv`）でも、全15条件で $Z_{\\perp} \\le 0.51$、$p \\ge 0.297$、BH-FDR補正後 $q = 1.000$（有意層 0/15）となった。すなわち、プローブ整列方向の消去は系統的な中和をもたらさず、直交ランダム方向を上回る出力変位も示さなかった。
- **Attention Pathway**: 単一層射影済みAttention出力の置換でも因果回復率は極小（Prompt最大1.48%, Generation最大4.23%）であり、tested single-layer projected Attention outputs did not show strong local causal recovery（本実験で検証した単一層の射影済みAttention出力が、強力な局所的因果ボトルネックとして機能する証拠は見出されなかった）。
- **Cross-Family Partial Replication**: Llama-3.2-1B-Instruct（11 valid pairs）での初期部分追試では生成時回復率は最大0.73%（pairwise中央値0.00%）に留まり、Qwenで観測された後段Residualへの明確な局在化は再現されなかった。これは因果局在のプロファイルがアーキテクチャや訓練目的に依存する可能性を示唆している。

したがって、

$$\\boxed{\\text{Linear Decodability} \\not\\Rightarrow \\text{Causal Localization}}$$

$$\\boxed{\\text{Decodability peak} \\neq \\text{Causal leverage peak}}$$

$$\\boxed{\\text{where information is decodable} \\neq \\text{where/when it becomes causally effective}}$$

すなわち、Prompt時において層別線形解読能と局所因果回復率の間に単調な相関は検出されず（$\\rho_{\\mathrm{MLP}}=0.296, \\rho_{\\mathrm{ATTN}}=0.023, \\rho_{\\mathrm{RESID}}=-0.039, p>0.05$）、最大解読能を示す中間層（L15 MLP）はPrompt時・Generation時を問わず局所因果レバレッジをほとんど持たない一方、強い因果レバレッジは自己報告生成直前の後段Residual Stream（L18〜L24）に時間的・空間的に解離して出現することが実証された。

---
"""

# Let's find the boundaries of this section
start_pattern = "*プローブ推定値の整合性に関する注記*:"
end_pattern = "### 6.2 Robustness to Likelihood Scoring"

start_idx = text.find(start_pattern)
end_idx = text.find(end_pattern)

if start_idx != -1 and end_idx != -1:
    print(f"Replacing corrupted section from {start_idx} to {end_idx}...")
    text = text[:start_idx] + correct_section_6_1_text + "\n" + text[end_idx:]
else:
    print(f"Could not find boundaries: start_idx={start_idx}, end_idx={end_idx}")

# Also update Abstract to the refined 3-pillar version
abstract_start_pattern = "語彙交絡を制御した最小対データセット"
abstract_end_pattern = "## 1. Introduction"

abstract_start_idx = text.find(abstract_start_pattern)
abstract_end_idx = text.find(abstract_end_pattern)

refined_abstract_text = """語彙交絡を制御した最小対データセット（AIPsy-Affect Strict Expanded: 192 pair-id groups / 422 samples）、81候補のValence–Arousal likelihood evaluation、全層linear probing、cross-model representation alignment、多変量OOD診断、およびwithin-model activation substitutionを統合した。

主要な発見は以下の3点に集約される：

第一に、affective peak versus neutral conditionの線形decodabilityは中盤層（Layer 14–15）で極大に達し、MLP outputで $R^2=0.561$、Attentionで $R^2=0.550$、Residual streamで $R^2=0.502$ を記録した。

第二に、この高い線形アクセス可能性は局所因果レバレッジを意味しない。初期の15-pair全層探索スクリーニング（Stage 1: exploratory screen）および全39組のHeld-out完全一致ペア（Prompt有効 N=32, Generation有効 N=39）を用いた代表層のfocused full-cohort evaluation（Stage 2）の双方において、decodability最高層（Layer 15 MLP）の真の2D Joint Optimal Transport回復率はPrompt-timeでわずか0.06%（中央値0.07%、全代表層最大でも1.08%）、Generation-timeでも1.39%（中央値0.50%）に留まった。層別decodabilityと因果回復率の間に単調相関は認められなかった（$\\rho \\le 0.296, p > 0.05$）。

第三に、強い因果レバレッジは自己報告生成時の後段Residual streamにおいて顕著に出現した（Layer 18: 41.48%, Layer 20: 50.22%, Layer 24: 53.48%）。一方、学習済みプローブ方向の幾何学的射影消去は系統的な中和をもたらさず、直交ランダム方向消去と統計的に区別されなかった（N=100高解像度追試で全層 $q = 1.000$）。なお、Meta Llama-3.2-1B-Instruct（11 valid pairs）に対する生成時部分追試では、Qwenで観測された後段Residualへの強い局所回復は再現されず（最大0.73%）、局所化プロファイルのモデル依存性が示唆された。

以上の結果は、**表現のアクセス可能性（representation accessibility）と因果的局所化（causal localization）が実証的に直交する別個の軸である**ことを示す。すなわち、線形プローブは情報がどこから読み取れるか（where information is readable）を同定するが、その情報がいつ・どこで下流計算に強い因果的影響力（causal leverage）を持つか、あるいはどの方向が必要であるかを指示しない。

---

"""

if abstract_start_idx != -1 and abstract_end_idx != -1:
    print("Updating Abstract...")
    text = text[:abstract_start_idx] + refined_abstract_text + text[abstract_end_idx:]

# Also update Limitations (Item 4)
text = re.sub(
    r"4\. \*\*因果スイープにおけるサンプル規模の制約\*\*:[^\n]+なお、重要代表層[^\n]+確認している。",
    "4. **因果スイープにおけるサンプル規模の制約**:\\n   全層×3コンポーネントに及ぶ全層網羅因果スクリーニング（Joint OT、生成時パッチング、プローブ方向射影消去）は、計算コスト制約から事前に固定したテストペアのサブセット（15 pairs）を用いて評価された（*The full-layer causal sweep used a computationally constrained subset of 15 held-out matched pairs; therefore, recovery estimates should be interpreted as localization evidence rather than precise population-level effect estimates.*）。なお、代表層（L10, 14, 15, 18, 20, 24）についてはHeld-out完全一致ペア（全39組評価; Prompt有効32ペア, Generation有効39ペア）による全数コホート評価を実施し、中盤層プローブピークにおけるPrompt-time局所十分性の低さ（~0%）および生成時における後段Residual Streamへの因果レバレッジの動的シフト（最大53.48%）を確認している。",
    text
)

# Also update Section 9 Conclusion
conclusion_pattern = r"## 9\. Conclusion[\s\S]+$"
correct_conclusion = """## 9. Conclusion

本研究は、大規模言語モデルにおける情動関連表現をケーススタディとして、内部表現の線形解読能（decodability）と、検証した局所介入における因果的レバレッジ（causal leverage）の所在との乖離を示した。

Qwen2.5-1.5Bを用いた体系的実験と、Llama-3.2-1B-Instructを用いたgeneration-time部分追試により、以下の包括的結論が得られた：

1. 中間層（Layer 14–15）において affective peak-versus-neutral condition は最も強く線形デコード可能であった（最大 $R^2 = 0.5610$）。
2. しかし、同一部位の局所活性化を置換しても、自己報告分布の回復率はプロンプト時でわずか 0.06%（Layer 15 MLP）、生成時でも 1.39% に留まり、線形解読能がピークとなる部位は局所的十分性を欠いていた。
3. 学習済みプローブ方向を除去しても系統的な中和傾向は認められず、その出力変位は直交ランダム方向の除去と統計的に区別されなかった（N=100高解像度追試でも全層 $q = 1.000$）。
4. 一方で、自己報告生成直前のコンテキストにおいては、後段Residual Stream（Layer 18〜24）に強い因果レバレッジが出現し、最大 53.48%（中央値 59.44%）の因果回復を達成した。

総じて、本研究の最終知見は次のように集約される：
> **Affective condition was most linearly decodable at intermediate layers, yet those same sites showed little causal recovery under matched activation substitution. Strong causal leverage instead emerged later, during report generation, particularly in the late residual stream. Probe-aligned direction removal showed no specific local necessity. Together, these results show that linear decodability identifies where information is accessible, but does not by itself identify where, when, or along which direction that information becomes causally effective.**

すなわち、本研究が実証した最終の中心命題は次のように定式化される：

$$\\boxed{\\text{Linear Decodability} \\not\\Rightarrow \\text{Causal Localization}}$$

$$\\boxed{\\text{where information is decodable} \\neq \\text{where/when it becomes causally effective}}$$

より一般的な解釈性研究の文脈において、本知見は次のように総括される：
> **“Linear decodability identifies where information is accessible, but does not by itself identify where, when, or along which direction that information becomes causally effective.”**

したがって、LLM内部表現をmechanisticに解釈する際には、probe accuracyやrepresentation similarityだけではなく、distributional diagnosticsと、prompt時および生成時双方を網羅したproperly validated causal interventionsを独立に組み合わせる必要がある。
"""

text = re.sub(conclusion_pattern, correct_conclusion, text)

# Write back cleanly encoded UTF-8
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully repaired {file_path}! Final length: {len(text)}")
