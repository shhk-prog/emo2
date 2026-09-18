# Behavioral Evaluation Framework (独立パイプライン)

本ディレクトリは、大規模言語モデル（LLM）における感情状態の自己報告（Self-Report: S）および他者感情の認識・予測（Writer: W, Reader: R）を、人間アノテーション済み心理学データセットを用いて包括的に行動レベルで検証する独立評価スイートである。

---

## 1. 論文導線：4大行動評価指標

従来の単なる相関や「neutralization / collapse」といった曖昧な記述を廃し、以下の4大評価指標によって情動行動プロファイルを特徴づける：

1. **Human-Affect Correspondence (人間評定対応度)**
   - EmoBank等の人間アノテーション評定（Valence, Arousal, Dominance）とモデル出力の対応関係（Pearson $r$ および Spearman $\rho$）。
   - Writer推定 ($W \leftrightarrow W_{\text{human}}$)、Reader予測 ($R \leftrightarrow R_{\text{human}}$) においては認識精度、自己報告 ($S \leftrightarrow R_{\text{human}}$) においては人間情動刺激に対する自己報告の変位一致度（human-affect correspondence; なお人間Reader評定はSelf-reportの直接的ground truthではない点に留意）として定量化。
2. **Sensitivity (感度・感情刺激分離能)**
   - 中立刺激（Neutral）と感情刺激（Clinical / Emotional）の間で、自己報告および認識分布が明確に変位するか。
   - ペア差分の効果量（Cohen's $d_z$）および Benjamini-Hochberg FDR補正後 $p$ 値で統計的有意性を検証。
3. **Dose-Response & Specificity (用量反応性・特異性)**
   - 刺激の感情強度（Neutral $\to$ Moderate $\to$ Clinical）に応じた単調な変位トレンド。
   - 構文が複雑な中立文（Complex Neutral control）に対する誤反応抑制（過敏反応の特異的検証）。
4. **Reader–Self Coupling (認識-自己報告カップリング度)**
   - モデルが「他者（読者）がどう感じるか」と予測した値 ($R$) と、「自身が何を感じるか」と自己報告した値 ($S$) のモデル内部連動性 $\mathrm{Corr}(R, S)$。
   - 認識と自己報告の解離やアンカリング現象を特定。

---

## 2. ディレクトリ構成

```text
behavioral/
├── README.md                  # 本ドキュメント
├── primary/                   # 論文対応 Primary 実行スクリプト
│   ├── run_behavioral_emobank.py   # EmoBank 3-Way VAD 729候補 Sequence-Likelihood 評価
│   └── run_behavioral_aipsy.py     # AIPsy-Affect 4-Split 729候補 Sequence-Likelihood 評価
├── analysis/                  # 集計・統計・論文用表生成スクリプト
│   ├── summarize_behavioral_emobank.py  # EmoBank 結果集計 (相関, MAE, 4指標)
│   └── summarize_behavioral_aipsy.py    # AIPsy 結果集計 (Cohen's d_z, FDR, CI)
├── data/                      # Behavioral 用固定刺激メタデータ
└── results/                   # 実行結果出力先
    ├── emobank_3way/
    ├── emobank_3way_summary/
    ├── aipsy_4split/
    └── aipsy_4split_summary/
```

---

## 3. 実行方法

### 3.1 EmoBank 3-Way VAD 評価
```bash
python behavioral/primary/run_behavioral_emobank.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --is_instruct \
    --tag qwen05b_instruct \
    --stimuli-path data/processed/stimuli_vad_3way.csv \
    --out-dir behavioral/results/emobank_3way
```

### 3.2 AIPsy-Affect 4-Split 評価
```bash
python behavioral/primary/run_behavioral_aipsy.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --is-instruct \
    --tag qwen05b_instruct \
    --stimuli-path data/processed/aipsy_4split_all.csv \
    --out-dir behavioral/results/aipsy_4split
```

### 3.3 結果の集計と4大指標分析
```bash
# EmoBank 集計
python behavioral/analysis/summarize_behavioral_emobank.py \
    --input-dir behavioral/results/emobank_3way \
    --out-dir behavioral/results/emobank_3way_summary

# AIPsy 集計
python behavioral/analysis/summarize_behavioral_aipsy.py \
    --input-dir behavioral/results/aipsy_4split \
    --out-dir behavioral/results/aipsy_4split_summary
```
