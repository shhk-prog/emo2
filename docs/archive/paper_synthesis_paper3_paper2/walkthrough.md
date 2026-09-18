# paper3.mdを基本としたpaper2.md統合・拡充の完了報告

ユーザーからのご指示「**基本はpaper3.mdの方として，足りない分の情報をpaper2からおぎなって詳しい論文にして**」に基づき、[`paper3.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md) の端正で厳密な章構成・文体を主軸としながら、[`paper2.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md) に含まれる中核図表・数学的定式化・全28層の完全数値アーカイブ・探索実験の詳細記録を余すところなく統合し、世界水準の決定版論文を完成させました。

---

## 1. 統合・拡充の主要ポイント

### (1) `paper3.md` の端正な骨格と学術的語り口の維持
- `paper3.md` が持つ「抑制的・客観的・学術的に厳密な文体」を厳格に維持。
- モデルに主観的感情を帰属させる表現を排し、操作的定義と測定されたエビデンス（線形アクセス可能性 vs 局所因果回復 vs プローブ方向特異性）に立脚した論理展開を徹底。

### (2) `paper2.md` から補完・統合された重要要素

1. **中核図表（4図＋3表）の完全な埋め込みと解説**:
   - **Figure 1**: 全体フレームワーク・概念図アスキーアートおよび中心仮説（$\text{Where information is decodable} \neq \text{where/when it becomes causally effective}$）。
   - **Figure 2**: 4-Panel 表現–因果プロファイル（(a) 解読能 $D_\ell$, (b) Prompt回復 $S_\ell$, (c) 必要性 $N_\ell$, (d) Generation回復 $G_{\ell,t}$）。
   - **Figure 3**: Direct Peak-Site Dissociation（最高解読部位 L15 MLP vs 最大因果回復部位 L24 Residual の直接対比サマリーおよび39ペア paired slope chart）。
   - **Figure 4**: Greedy Collapse vs. Distributional Sensitivity（(a) Greedy (5,5) 崩壊率 98.6% vs 12.4%, (b) 人間Valence vs 期待値 $E[V]$ 散布図 $r=0.629$ vs $r=0.365$）。
   - **Table 1**: 実験設計およびデータセット諸元（モデル、層数、AIPsy-Affect Strict Expanded構成、81候補列、2D Joint OT）。
   - **Table 2**: 39完全テストペアにおける代表サイト結果サマリー（代表6サイトの $D_\ell, S_\ell, G_{\ell,t}$ と 95% Bootstrap CI）。
   - **Table 3**: 研究の問い・方法論的証拠・実証結果・理論的示唆の総括対照表（Claims and Evidence）。

2. **数学的・方法論的定式化の厳密化**:
   - 81候補列拘束系列対数尤度 $s^{\mathrm{raw}}_{v,a}(x)$、長さ正規化スコア $s^{\mathrm{norm}}_{v,a}(x)$、期待値 $E[V], E[A]$ の定義式。
   - 2次元 Optimal Transport（Wasserstein-2 距離、Manhattan ground cost）の数学的定義式。
   - OT分母セーフガード閾値（$\mathrm{OT}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < 0.05$）の統計的根拠と有効ペア数（Prompt時32/39、Gen時39/39）。
   - プローブ方向の幾何学的直交射影消去式、50/100本の直交帰無分布生成法、標準化特異性 $Z_\perp$ と FDR 補正。

3. **体系的Appendix（再現性の保管庫）の完全収録**:
   - **付録A**: データセット・分割完全性プロトコル
   - **付録B**: 81候補列尤度測定・プロンプト全文
   - **付録C**: 活性化抽出フック・プローブ推定詳細
   - **付録D**: Matched-Substitution プロトコル詳細
   - **付録E (Table A1)**: 全28層 Prompt-Time スクリーニング完全数値表（全84サイトの $D_\ell, S_\ell$ 網羅）および Marginal OT 頑健性
   - **付録F (Table A2, Table A5)**: 全28層 Generation-Time スクリーニング完全数値表（全84サイトの $G_{\ell,t}$ 網羅）および 代表18サイト39ペア詳細統計・IQR・95% CI
   - **付録G (Table A3, Table A4)**: 84サイト プローブ方向除去必要性検定完全数値表（全84サイト $Z_\perp$, FDR $q_\perp$）および N=100高解像度追試
   - **付録H (Table A6)**: 多層Residual同時介入の飽和特性データ（1層〜7層連続ブロック）
   - **付録I (Table A7)**: Base-to-Instruct Ridge写像・正則化スイープ・マハラノビスOOD診断数値表
   - **付録J**: 代替アウトカムと探索実験（系列全体置換の破壊性、気分一致性ステアリング3,210観測の詳細）
   - **付録K**: LLaMA-3.2-1B-Instruct クロスファミリー追試完全記録
   - **付録L (Table A8)**: 再現スクリプト・生成データ・ハッシュ対応マニフェスト

---

## 2. 完成した決定版論文のファイル構造

- **完成論文原稿**: [`v3/docs/paper3.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md)（計751行、約65KBの決定版原稿）
- **原本保持**: [`v3/docs/paper2.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md)（変更せず原本として保持）
- **プロット生成スクリプト**:
  - 本文主図スクリプト: [`v3/scripts/plot_paper_figures.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_paper_figures.py)
  - 付録図スクリプト: [`v3/scripts/plot_appendix_figures.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_appendix_figures.py)

---

## 3. プロジェクトドキュメント保存確認

プロジェクトドキュメント保存ルールに基づき、本トピックに関するドキュメントは以下の場所に保存されています：
- タスクリスト: [`task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_synthesis_paper3_paper2/task.md)
- 実装計画: [`implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_synthesis_paper3_paper2/implementation_plan.md)
- 完了報告: [`walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_synthesis_paper3_paper2/walkthrough.md)
