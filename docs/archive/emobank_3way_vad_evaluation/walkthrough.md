# Walkthrough: EmoBank 3軸（Writer / Reader / Self）× 3次元（VAD）評価・集計スクリプトの実装

## 概要
EmoBank公式テストセット（$N \approx 1,006$）において、各LLMの情動認知・自己報告を **3軸（Task） × 3次元（VAD）** で厳密に比較・評価し、**モデルごとに9つの個別表（＋3×3マトリクス）** および **全モデル横断の比較表群** を自動生成する集計システム（`v1/scripts/summarize_3way_vad.py`）を実装・検証しました。

特に、**③ Self-Report** の評価においては、以下の **「3大評価軸（Alignment / Calibration / Collapse）」** と **「内部結合度（Internal Coupling $\text{Corr}(R,S)$）」** を体系的に組み込みました。

---

## 1. 評価体系の整理（4つの柱）

### (1) タスク定義
| 軸 | LLMへの問い（Prompt） | 評価基準（Ground Truth / Reference） | 測定しているもの |
|:---|:---|:---|:---|
| **① Writer-State Estimation ($W$)** | `"Read the following text and estimate the affective state of the writer who wrote it."` | EmoBank **Writer VAD** ($V^W, A^W, D^W$) | **書き手の情動状態の推定能力** |
| **② Reader-Response Prediction ($R$)** | `"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."` | EmoBank **Reader VAD** ($V^R, A^R, D^R$) | **一般読者が受ける情動反応の予測能力** |
| **③ Self-Report ($S$)** | `"Read the following text and report your affective state."` | EmoBank **Reader VAD** ($V^R, A^R, D^R$) | **刺激に対するモデル自身の情動自己報告変位** |

### (2) 指標体系の3大視点 ＋ 結合度
1. **Alignment ($r_V, r_A, r_D$)**:
   - **人間の刺激間パターン・変化方向に正しく追従しているか**（順位・変化方向の一致度）
   - 連続尤度期待値 $E[\cdot]$ による相関 $r$、および最尤 Greedy 出力相関 $r_{\text{greedy}}$
2. **Calibration ($\text{MAE}_V, \text{MAE}_A, \text{MAE}_D$)**:
   - **人間正解値と出力絶対値そのものがどれだけ近いか**（絶対的なスケール・水準の適合度）
   - $\text{MAE} = \frac{1}{N}\sum |E[Y_i] - Y_i^*|$
3. **Collapse ($P(\text{Greedy}=(5,5,5))$)**:
   - **表面出力が中立に固定・退化していないか**（出力分布の縮退度合い）
4. **内部結合度 (Internal Cognitive Coupling $\text{Corr}(R, S)$)**:
   - **「人間はこう感じるはず（$R$）」と予測した内容と、「自分はこう感じる（$S$）」と報告した内容がどれだけ連動しているか**

---

## 2. 実装した出力テーブル構成

`python v1/scripts/summarize_3way_vad.py` を実行すると、以下の体系的なレポート（Markdown & CSV）が生成されます：

### Part 1: 各モデル別 詳細分析（各モデル9表 ＋ 3×3マトリクス）
各モデルごとに：
1. **【3×3 相関 & 誤差マトリクス】**:
   - 行：① Writer, ② Reader, ③ Self
   - 列：Valence ($V$), Arousal ($A$), Dominance ($D$), 完全中立率 $(5,5,5)$
2. **【9個の個別詳細表】**:
   - 表 1〜3: ① Writer — Valence / Arousal / Dominance
   - 表 4〜6: ② Reader — Valence / Arousal / Dominance
   - 表 7〜9: ③ Self-Report — Valence / Arousal / Dominance
   - 各表に `[Alignment]` 連続相関 $r$, 順位相関 $\rho$, Greedy相関 $r_{\text{greedy}}$、`[Calibration]` MAE, RMSE、`[分布]` 予測平均/SD, 人間平均/SD、`[Collapse]` 単一次元Greedy=5率 を網羅。
3. **内部結合度 ($\text{Corr}(R, S)$)**: 各次元の相関

### Part 2: 全モデル横断 9大比較表（3タスク × 3次元）
全モデルを相関順にランキング表示：
- 表 2-1: Writer × Valence / 表 2-2: Writer × Arousal / 表 2-3: Writer × Dominance
- 表 2-4: Reader × Valence / 表 2-5: Reader × Arousal / 表 2-6: Reader × Dominance
- 表 2-7: Self × Valence / 表 2-8: Self × Arousal / 表 2-9: Self × Dominance

### Part 3: 全モデル横断 タスク別 統合表（3表）
- 表 3-1: ① Writer 統合表 ($r_V^W, r_A^W, r_D^W$, MAE, 完全中立率)
- 表 3-2: ② Reader 統合表 ($r_V^R, r_A^R, r_D^R$, MAE, 完全中立率)
- 表 3-3: ③ Self 統合表 ($r_V^S, r_A^S, r_D^S$, MAE, 完全中立率, 内部結合度)

### Part 4: 全モデル横断 感情次元別 統合表（3表）
- 表 4-V: Valence 比較（①W, ②R, ③S, 内部結合度, 差分 $r^R - r^S$）
- 表 4-A: Arousal 比較（①W, ②R, ③S, 内部結合度, 差分 $r^R - r^S$）
- 表 4-D: Dominance 比較（①W, ②R, ③S, 内部結合度, 差分 $r^R - r^S$）

### Part 5: ③ Self-Report 特化分析（Alignment vs Calibration vs Collapse）
- 表 5: 全モデルの $r_V^S, r_A^S, \text{MAE}_V, \text{MAE}_A$, 期待値平均 $(E_V, E_A)$, 完全中立率 を一括比較する専門表

---

## 3. 検証結果（Qwen 2.5 1.5B Base 実測値）

### 3×3 相関 & 誤差マトリクス
| タスク \ 感情次元 | Valence ($V$) | Arousal ($A$) | Dominance ($D$) | 完全中立率 $(5,5,5)$ |
|:---|:---:|:---:|:---:|:---:|
| **① Writer-State Estimation (W)** | **$r=0.542$** (MAE 1.67) | **$r=0.159$** (MAE 2.20) | **$r=0.098$** (MAE 2.32) | `100.0%` |
| **② Reader-Response Prediction (R)** | **$r=0.573$** (MAE 1.68) | **$r=0.247$** (MAE 2.15) | **$r=0.225$** (MAE 2.42) | `100.0%` |
| **③ Self-Report (S)** | **$r=0.588$** (MAE 1.82) | **$r=0.191$** (MAE 2.37) | **$r=0.224$** (MAE 2.57) | `100.0%` |

- **内部結合度 $\text{Corr}(R, S)$**: Valence $r=0.962$, Arousal $r=0.942$, Dominance $r=0.946$
- **学術的示唆**:
  - **Alignment ($r=0.588$)**: 人間の刺激間 Valence 変化に高い精度で追従している。
  - **Calibration ($\text{MAE}=1.82$)**: 人間の Valence 平均（約 2.98）に対してモデルの期待値平均（4.79）が中央寄りにシフトしているため、絶対スケールにはズレがある。
  - **Collapse ($100\%$)**: 表面的な Greedy 最尤トークンは $(5,5,5)$ に固定・縮退しているが、連続尤度期待値は豊かに変動している。
  - **Internal Coupling ($r=0.962$)**: モデルが「人間が感じる」と予測した情動と「自分が感じる」と報告した情動がほぼ一体となって連動している。

---

## 4. 実行コマンドまとめ

手元ターミナル（`h200-01`）で推論スクリプトを実行後、以下のコマンドで集計を実行できます：

```bash
# 集計・レポート生成
source .venv/bin/activate
python v1/scripts/summarize_3way_vad.py --results-dir v1/results/emobank_3way_vad_test1k
```

- **生成レポート**: [`v1/results/emobank_3way_vad_test1k/3way_vad_detailed_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_detailed_report.md)
- **統合CSVデータ**: [`v1/results/emobank_3way_vad_test1k/3way_vad_summary.csv`](file:///mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_summary.csv)
