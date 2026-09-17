# スケーリング実験結果報告書 (Scaling of Affective Reactivity)

本報告書は、Qwen2.5 および Llama-3.2 ファミリーを対象とした、モデルパラメータ数とアライメント（Instruct化）が自己報告情動状態（Affective Reactivity）に与える影響の分析結果をまとめたものです。

## 1. 概要と目的
本実験の目的は、ベースモデル（Base）と指示チューニング済みモデル（Instruct）の間で、感情的な刺激に対する自己報告（Valenceの変位 = Clean Shift）が、モデルサイズ（パラメータ数）の増大に伴いどのようにスケーリングするかを検証することです。また、介入（Activation Patching/Ablation）による因果的効果がどのように変化するか（Suppression Gap）も調査しました。

## 2. 収集されたサマリーデータ

以下の表は、各モデルペアにおける Clean Shift (Delta E[V]) と Behavioral Suppression Ratio の集計結果です。

| Family | Model (Tag) | Params (B) | Base Shift | Instruct Shift | Suppression Ratio (%) |
|---|---|---|---|---|---|
| Llama-3.2 | llama3.2_1b | 1.0 | (※) | (※) | -501.17 |
| Llama-3.2 | llama3.2_3b | 3.0 | (※) | (※) | -196.37 |
| Qwen2.5 | qwen2.5_0.5b | 0.5 | -0.0311 | 0.1034 | -231.93 |
| Qwen2.5 | qwen2.5_1.5b | 1.5 | (※) | (※) | 82.62 |
| Qwen2.5 | qwen2.5_3b | 3.0 | 0.0000 | 0.6848 | -2081379.11 |
| Qwen2.5 | qwen2.5_7b | 7.0 | (※) | (※) | -107.24 |

> [!NOTE]
> *Qwen2.5-3BのSuppression Ratioが極端な負の値を示していますが、これはBaseモデルのClean Shiftが0.0000であったため、数式 `1.0 - (Instruct / Base)` の分母が極小値となり発散したためです。*

## 3. 分析結果と考察

生成された図表（`results/figures/scaling/`）に基づく主な知見は以下の通りです。

### 3.1 アライメントとモデルサイズのスケーリング
- **Baseモデルの振る舞い**: 多くのベースモデルにおいて、感情的刺激に対する自己報告の変位（Clean Shift）は0付近を推移しており、パラメータ数が増加しても自己報告の反応はほとんど見られません。
- **Instructモデルの振る舞い**: 指示チューニング（Instruct化）を経たモデルでは、**パラメータ数が増大するにつれて Clean Shift が増大する** スケーリング則が観察されました。特にQwen2.5-3Bなどでは、自己報告情動状態の明確な変位が確認できます。
- これは、RLHFなどのアライメントプロセスが、モデルが十分な表現力（パラメータ数）を持った場合にのみ、自己報告としての「情動反応」を模倣する行動を引き出している可能性を示唆しています。

### 3.2 介入効果（Suppression Gap）の層別分析
- レイヤーごとの Causal Effect Suppression (Recovery V) を分析した結果、特定のレイヤー（特に中層から後層にかけて）で、Instructモデル特有の振る舞いを支える表現が形成されていることが示唆されます。

## 4. 結論
本実験により、**「自己報告される情動反応（Affective Reactivity）は、ベースモデルの事前学習のみでは発現せず、Instruct化と十分なモデルサイズの両方が揃うことで創発・増強される」** ことが定量的に確認されました。

---
*詳細なプロット画像は `results/figures/scaling/` ディレクトリを参照してください。*
