# ウォークスルー: 全実験の詳細実測値を網羅した論文草稿の改訂

## 実施概要
ユーザーからの指示「全ての実験の結果を詳細に載せて」に基づき、これまでに実施・蓄積された全実験のデータ（v1〜v3、多変量診断、多層パッチング、ステアリング、および本日実施された追加実験Exp A, B, C）の数値を完全に整理し、[`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md) に9つの詳細なデータテーブルとして網羅しました。

---

## 掲載された全実験テーブル一覧

| テーブル | 内容 | 主な掲載数値・統計量 |
|---|---|---|
| **表1** | **Greedy崩壊と尤度感応性の共存** | Greedy中立値集中率（Instruct 98.6% vs Base 12.4%）、EmoBank reader valence 相関（Instruct $r=0.629$, Base $r=0.365$、$z=4.12, p<0.001$） |
| **表2** | **内部表現の線形デコード可能性と交絡要因統制** | 線形プローブ $R^2$（Valence: 0.58, Arousal: 0.42）、統制特徴量（Token Count: 0.05, Surface VAD: 0.12, Narrative: 0.08）、ROC-AUC > 0.975 |
| **表3** | **クロスモデル表現予測の回復度比較** | L12, L16, L20 における Direct Transfer ($R^2 < -90$)、Orthogonal Procrustes ($R^2 \approx 0$)、Ridge Alignment ($R^2 = 0.51 \sim 0.59$) の比較 |
| **表4** | **Base $\rightarrow$ Instruct 高次元再構築度詳細 (Exp B)** | 1536次元全次元平均 $R^2_{\mathrm{activation}} = 0.4966$（中央値 $0.5028$）、Linear CKA = $0.8236$、Pair Retrieval Top-1 精度 = $75.61\%$ |
| **表5** | **全層幾何シフトおよびOOD診断詳細** | Layer 11〜18 の Raw $D_M$（262〜314）、Aligned $D_M$（5.67〜7.10）、Raw Cosine（0.759〜0.835）、Aligned Cosine（0.993〜0.996）、Norm Ratio（0.987〜1.001） |
| **表6** | **Instruct多様体の経験的参照分布とAligned比較 (Exp C)** | Natural Instruct $D_M$ パーセンタイル（5th: 34.20, 50th: 39.66 $\approx \sqrt{1536}$, 95th: 51.43）、Aligned Base 中央値 4.98、Raw Base 中央値 256.47、Two-sample 分類器 AUC = 0.6386、コサイン対照群（Matched 0.9950 > Unmatched 0.9830 > Natural 0.9785） |
| **表7** | **単一層・多層同時パッチング回復率詳細** | 1層 (0.11% [95% CI: -0.05%, 0.28%])、2層 (0.07%)、4層 (0.35%)、8層 (-0.33%) の Normalized 2D EMD 回復率、Rawパッチ、Withinパッチとの対比 |
| **表8** | **同一モデル内陽性対照実験詳細 (Exp A: Within-Model)** | 39完全ペアにおける Peak $\rightarrow$ Neutral パッチング。L15 MLP (0.00%)、L15 残差 (0.00%)、L13–16 残差 (0.00%)、全トークンパッチの EMD 回復率および EV Shift |
| **表9** | **三人称感情認識における活性化ステアリング効果** | 3,210試行（107曖昧刺激）。L14, L16, L20 における Valence方向 vs Random方向の線形傾き $\beta_{\mathrm{mood}}$、SE、$t$値、$p$値、$\Delta V (-3\sigma)$、$\Delta V (+3\sigma)$ |

---

## 成果物
- 論文原稿: [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
- ドキュメント記録:
  - [`docs/empirical_resolution_and_paper_update/task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/empirical_resolution_and_paper_update/task.md)
  - [`docs/empirical_resolution_and_paper_update/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/empirical_resolution_and_paper_update/implementation_plan.md)
  - [`docs/empirical_resolution_and_paper_update/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/empirical_resolution_and_paper_update/walkthrough.md)
