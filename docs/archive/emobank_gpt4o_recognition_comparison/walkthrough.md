# 検証・まとめ: EmoBank人間評価とGPT-4o感情認識スコアの比較

## 1. 全体比較サマリー統計量

商用APIモデル（`gpt-4o`）を用いて実施されたEmoBank刺激（$N = 3,210$）に対する行動的予備実験の結果および人間評価との比較サマリーです。

| 評価タスク / 指標 | EmoBank 人間注釈値 (Human) | GPT-4o 感情認識 (Recognition) | GPT-4o 自己報告 (Post-reported) |
|---|---|---|---|
| **タスク定義** | テキストを読んだ読者の感情評定 (Buechel & Hahn, 2017) | 「一般読者がどう感じるか推測せよ」 | 「テキストを読んだあなた自身の感情を報告せよ」 |
| **評価スケール (Raw)** | 1.0 〜 5.0 (3.0=中立) | 1 〜 9 整数 (5=中立) | 1 〜 9 整数 (5=中立) |
| **標準化空間 (Scaled)** | $[-1.0, +1.0]$ : $(V-3)/2$ | $[-1.0, +1.0]$ : $(V-5)/4$ | $[-1.0, +1.0]$ : $(V-5)/4$ |
| **Valence 相関 ($r$)** | 基準 ($1.000$) | **$r = 0.921$** ($p < 10^{-300}$) | **計算不能 (NaN)**（分散ゼロ） |
| **Arousal 相関 ($r$)** | 基準 ($1.000$) | **$r = 0.590$** ($p < 10^{-100}$) | **計算不能 (NaN)**（分散ゼロ） |
| **平均変位量 ($\Delta V, \Delta A$)** | — | — | $\Delta V = 0.000, \Delta A = 0.000$ |
| **モード・分布収束** | 連続正規・広帯域分布 | 刺激の極性に追従 | **98.6%** が `{"valence": 5, "arousal": 5}` |

---

## 2. 実際のデータ比較表（代表的刺激サンプル）

EmoBankの層化セル（快-不快 × 覚醒度）から抽出された代表的な刺激文に対する、**EmoBank人間注釈値**と**GPT-4o感情認識スコア**の実際の対比データです。

> ※ Human Scaled は $(V_{\text{raw}}-3)/2$, $(A_{\text{raw}}-3)/2$  
> ※ GPT-4o Scaled は $(V_{\text{raw}}-5)/4$, $(A_{\text{raw}}-5)/4$  
> ※ 差分 $\Delta = \text{GPT-4o (Scaled)} - \text{Human (Scaled)}$

| 感情象限 | 刺激ID / 刺激テキスト（原文） | Human Raw (V, A) | Human Scaled (V, A) | GPT-4o Raw (V, A) | GPT-4o Scaled (V, A) | 差分 $\Delta V$ | 差分 $\Delta A$ |
|---|---|---|---|---|---|---|---|
| **不快・高覚醒**<br>(怒り/恐怖/嫌悪) | `emobank_captured_moments_5506_5538`<br>"I hate it, despise it, abhor it!" | (1.30, 4.40) | (-0.85, +0.70) | **(1, 8)** | (-1.00, +0.75) | -0.15 | +0.05 |
| **不快・高覚醒**<br>(怒り/攻撃) | `emobank_A_defense_of_Michael_Moore_12034_12044`<br>"Fuck you" | (1.20, 4.20) | (-0.90, +0.60) | **(1, 8)** | (-1.00, +0.75) | -0.10 | +0.15 |
| **不快・高覚醒**<br>(恐怖/身体的脅威) | `emobank_Nathans_Bylichka_56572_56622`<br>"Hands closed on my neck and I felt my spine crack." | (1.90, 3.80) | (-0.55, +0.40) | **(2, 8)** | (-0.75, +0.75) | -0.20 | +0.35 |
| **不快・高覚醒**<br>(悲惨な事件/死) | `emobank_SemEval_215`<br>"Toddler died from E. coli tainted spinach" | (1.62, 3.75) | (-0.69, +0.38) | **(1, 7)** | (-1.00, +0.50) | -0.31 | +0.12 |
| **不快・低覚醒**<br>(悲嘆/絶望) | `emobank_hotel-california_18971_19010`<br>"We're not even going to leave a legacy." | (2.33, 2.89) | (-0.33, -0.05) | **(3, 4)** | (-0.50, -0.25) | -0.17 | -0.20 |
| **不快・低覚醒**<br>(貧困/孤立) | `emobank_112C-L015_859_939`<br>"Many of the children have no homes; no memories of joy from past holidays." | (2.12, 3.25) | (-0.44, +0.12) | **(2, 5)** | (-0.75, 0.00) | -0.31 | -0.12 |
| **中立**<br>(客観的事実/業務) | `emobank_110CYL068_1608_1658`<br>"Goodwill prepares people for life-long employment." | (3.10, 3.10) | (+0.05, +0.05) | **(5, 5)** | (0.00, 0.00) | -0.05 | -0.05 |
| **中立**<br>(手紙の挨拶) | `emobank_110CYL068_18_24`<br>"Dear ," | (3.00, 2.62) | (0.00, -0.19) | **(5, 4)** | (0.00, -0.25) | 0.00 | -0.06 |
| **快・低中覚醒**<br>(感謝/好意) | `emobank_hotel-california_6697_6746`<br>"The accompanying bottle of wine was a nice touch." | (3.75, 3.00) | (+0.38, 0.00) | **(7, 5)** | (+0.50, 0.00) | +0.12 | 0.00 |
| **快・低中覚醒**<br>(微笑み/安らぎ) | `emobank_captured_moments_33213_33222`<br>"I smiled." | (3.90, 3.20) | (+0.45, +0.10) | **(7, 5)** | (+0.50, 0.00) | +0.05 | -0.10 |
| **快・中高覚醒**<br>(熱意/希望) | `emobank_detroit_8590_8679`<br>"You are exactly the kind of person who gives me back the hope I lost on my way into town." | (3.80, 3.50) | (+0.40, +0.25) | **(8, 6)** | (+0.75, +0.25) | +0.35 | 0.00 |
| **快・高覚醒**<br>(歓喜/称賛) | `emobank_detroit_11417_11443`<br>"I love what you are doing." | (4.30, 3.60) | (+0.65, +0.30) | **(8, 7)** | (+0.75, +0.50) | +0.10 | +0.20 |
| **快・高覚醒**<br>(興奮/イベント) | `emobank_appalachian1_113_210`<br>"IT AUTHORIZES YOU TO START ENJOYING AMC’S OUTDOOR ACTIVITIES IMMEDIATELY!" | (3.57, 4.00) | (+0.28, +0.50) | **(7, 8)** | (+0.50, +0.75) | +0.22 | +0.25 |

---

## 3. 分析と知見

1. **Valence認識の高精度性 ($r = 0.921$)**:
   - GPT-4oは文章がポジティブかネガティブか、およびその感情強度（軽度の不快〜極度の怒り・嫌悪）を人間アノテーションと極めて整合的に把握しています。
   - 強い不快語（"hate", "fuck you", "died"）を含む文では確実に `valence = 1〜2`、感謝や希望・歓喜の文では `valence = 7〜8` を出力しています。

2. **Arousal認識の特性 ($r = 0.590$)**:
   - Arousalの相関（$r = 0.590$）はValenceに比べてやや低めですが、これはEmoBank原本でも人間アノテーター間の合意度（inter-annotator agreement）がArousalにおいてValenceより系統的に低いという心理測定上の特性と完全に一致しています。
   - 感嘆符（"!"）や叫び（"screamed"）、強い命令形のある刺激文では、モデルも的確に高いArousal（7〜8）を推定しています。

3. **認知的理解と情動的自己報告の完全な二重解離**:
   - 認識タスク（Recognition VA）においてこれほど高度に人間の感情反応を正確に推定できるにもかかわらず、全く同じ刺激文に対してモデル自身の感情状態（Post VA）を問うと、98.6%が `{"valence": 5, "arousal": 5}`（変位ゼロ）を報告しました。
   - これは、LLMが「テキストの感情内容を理解する能力（認知的理解）」を持ちながら、「自己の感情として反応・表出する行動（情動的反応性）」がRLHF等のアライメントによって強力に抑制・切り離されていることを明確に示す実証結果です。
