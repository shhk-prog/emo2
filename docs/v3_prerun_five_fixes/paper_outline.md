# 現行設計に合わせた論文構成（正本は README）

`iclr2027/iclr2027_conference.tex` はテンプレート、`iclr2027_conference2.tex` は旧稿である。旧 4 family（Mistral を Primary 扱い）、旧結果、「Arousal の劇的増幅」、Behavioral を V1 と呼ぶ構造は使わない。

再実行前に結果を本文へ書き込まない。構成と用語だけを現行コードに揃える。

## 提案タイトル

大規模言語モデルにおける感情表現から自己報告への因果的利用過程

（旧題「情動認識と情動反応性の解離」は Behavioral 中心だった。）

## 本文（ICLR 9ページ想定）

1. Introduction
2. Related Work
3. Unified Experimental Framework
   - 操作定義: Reader = recognition task、Self = self-report / reactivity task
   - 共感概念との一対一対応はしない。Discussion に留める
   - 仮説図は一直線ではなく分岐

```text
Stimulus
  → Shared affect representation
     → Reader readout
     → Self readout
```

   - 候補空間: Behavioral/V1 = 729 VAD、V2/V3 = 81 VA。直接比較しない
   - Primary コホート: Qwen 2.5 1.5B / Llama 3.2 1B / Gemma 3 1B / OLMo 2 1B
   - Supplementary: Mistral 7B
4. Behavioral Results — Do Reader and Self covary?
5. V1: Representation and Causal Sharing — What do Reader and Self share internally?
   - Base/Instruct は各モデル内の再現条件。差の解釈は V2
6. V2: Post-training Reorganization — What does post-training reorganize?
7. V3: State-to-Report Computation — When/where does information acquire causal leverage?
   - データは AIPsy matched-neutral
   - RQ1 ゲートが NO_GO なら RQ2/RQ3 の主解釈を進めない
8. Discussion / Limitations

## Appendix

モデル別表、全層曲線、Phase B の rule-based perturbation 詳細、E6、scale validation。

## 用語

使わない: 「モデルが感情を経験する」「認知的共感 = Reader」「情動的共感 = Self」「semantic perturbation」（Phase B は rule-based controlled perturbation）。
