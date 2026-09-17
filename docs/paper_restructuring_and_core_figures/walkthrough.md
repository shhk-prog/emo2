# 論文本文の再構成（4図＋3表）および再現性アーカイブ整備の完了報告

ユーザーからの「読者が遭難しないよう、中心命題である『Decodability peak と causal leverage peak が空間的・時間的に解離する』ことを一目で理解させる図を中心に本文を再編し、詳細な全層結果・頑健性・探索的実験をAppendixへ送る」という設計方針に基づき、以下の再構成・スクリプト作成・ドキュメント整備を完了しました。

---

## 1. 成果物の全体概要

1. **本文用図表生成スクリプト**: [`plot_paper_figures.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_paper_figures.py)
   - **Figure 1**: Overall Experimental Framework（概念図・中心仮説ブロック）
   - **Figure 2**: Four-Panel Representational–Causal Profile（主図：$D_\ell, S_\ell, N_\ell, G_{\ell,t}$ の層別対比）
   - **Figure 3**: Direct Peak-Site Dissociation（L15 MLP vs L24 Residual の直接対比・39ペア paired 可視化・$\Delta G = +53.30\%$ [45.34, 61.16]）
   - **Figure 4**: Greedy Collapse vs Distributional Sensitivity（Base 12.4% vs Instruct 98.6% の (5,5) 集中と $E[V]$ 相関 scatter）
2. **Appendix用図表生成スクリプト**: [`plot_appendix_figures.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_appendix_figures.py)
   - Fig A1〜A10（全28層プロファイル、アブレーション詳細、Ridgeスイープ、Mahalanobis診断、多層飽和、LLaMA追試等）
3. **本文用 3つの表の定式化**:
   - **Table 1**: Dataset and Experimental Design（実験諸元・データセット構成・評価指標）
   - **Table 2**: Main Representative-Site Results（代表6サイトの $D_\ell, S_\ell, G_{\ell,t}$ と 95% Bootstrap CI）
   - **Table 3**: Synthesis of Core Claims and Empirical Evidence（問い・エビデンス・結論の総括対照表）
4. **論文本文新稿の作成と `paper2.md` への統合**:
   - [`paper_restructured.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper_restructured.md) として独立原稿を作成。
   - さらに、ユーザーからのご指示に基づき、[`paper2.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md) の末尾に「第II部 統合再編：中核『4図＋3表』による明快な論文プレゼンテーション」および「第III部 体系的再現性付録アーカイブ」として日本語で完全統合・追記。
5. **完全な再現性保管庫（Appendix）**: [`appendix_tables_and_figures.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/appendix_tables_and_figures.md)
   - Table A1（全28層Prompt値）、Table A2（全28層Generation値）、Table A3（84サイトアブレーション検定値）、Table A5（18サイト39ペア詳細）、Table A6（多層介入）、Table A7（Ridgeスイープ）、Table A8（再現スクリプト対応表）を網羅。

---

## 2. 本文の「4図＋3表」構成

| 番号 | 形式 | タイトル | 論文中での役割 |
|---|---|---|---|
| **Fig. 1** | 概念図 | Overall Experimental Framework & Central Finding | 研究全体のパイプライン（Prompt-time vs Generation-time）と、中心仮説「Where information is decodable $\neq$ where/when it becomes causally effective」を提示。 |
| **Fig. 2** | 4パネル図 | Four-Panel Representational–Causal Profile | **論文の主図**。(a) 線形解読能 $D_\ell$（L15 MLP peak: 0.561）、(b) Prompt介入回復率 $S_\ell$（$\approx 0.5\%$）、(c) プローブ方向必要性 $N_\ell$（全層特異性なし）、(d) 生成時介入回復率 $G_{\ell,t}$（L24 Residual: $53.24\%$）を同一の横軸（0–27層）で対比。 |
| **Fig. 3** | 対比・散布図 | Direct Peak-Site Dissociation | **統計的決定打**。L15 MLP（$D=0.561, G=-0.06\%$）と L24 Residual（$D=0.147, G=53.24\%$）の39ペア全数 paired slope chart。$\Delta G = +53.30\%$, 95% CI: $[+45.34\%, +61.16\%]$, $p < 10^{-12}$ を明示。 |
| **Fig. 4** | 棒グラフ＋散布図 | Greedy Collapse vs. Distributional Sensitivity | RQ1の解決。(a) Greedy (5,5) 集中率（Base 12.4% vs Instruct 98.6%）、(b) 人間Valence vs 期待値 $E[V]$（Base $r=0.365$ vs Instruct $r=0.629$）。Greedy Collapse $\neq$ Distributional Invariance を一撃で提示。 |
| **Table 1** | 表 | Dataset and Experimental Design Specifications | モデル、層数、AIPsy-Affect Strict Expanded（192 groups / 422 samples）、81候補列尤度評価、2D Joint OT距離など方法論の諸元。 |
| **Table 2** | 表 | Main Representative-Site Results across 39 Test Pairs | 代表6サイト（L14 Resid, L15 MLP, L18 Attn, L18 Resid, L20 Resid, L24 Resid）の $D_\ell, S_\ell, G_{\ell,t}$ と 95% Bootstrap CI。 |
| **Table 3** | 表 | Synthesis of Core Claims and Empirical Evidence | 読者が「結局何を示したのか」を瞬時に把握できる総括表（RQ1〜RQ5、エビデンス、結論）。 |

---

## 3. 本文とAppendixの明確な棲み分け

- **本文（約8ページ相当のコンパクトで強固な構成）**:
  - RQ1: 尤度感度とGreedy集中の解離（Fig. 4）
  - RQ2: 中間層でdecodableだがPrompt介入回復はゼロ・アブレーション特異性なし（Fig. 2 a–c, Table 2）
  - RQ3: 生成直前Residualへの因果性シフトと2大ピークの統計的解離（Fig. 2 d, Fig. 3, Table 2）
  - 要約段落: 多層介入飽和（55%で頭打ち）、Ridgeアライメントの多様体崩壊、LLaMA追試
  - 総括: Claims and Evidence（Table 3）、方法論的・概念的示唆
- **Appendix（再現性の保管庫）**:
  - Appendix A: 全28層 Prompt-time スイープ数値表（Table A1; Fig A1, A2, A3）
  - Appendix B: 全28層 Generation-time スイープ数値表（Table A2; Fig A4）
  - Appendix C: 84条件 プローブ方向除去検定とFDR補正値（Table A3, A4; Fig A5, A6）
  - Appendix D: 39ペア 代表18サイト詳細統計・IQR・CI（Table A5）
  - Appendix E: 多層同時介入の飽和特性（Table A6; Fig A9）
  - Appendix F: Ridge正則化スイープとMahalanobis OOD診断（Table A7; Fig A7, A8）
  - Appendix G: LLaMA-3.2-1B-Instruct 追試プロファイル（Fig A10）
  - Appendix H: 再現スクリプト・生成アーティファクト・ハッシュ対応表（Table A8）

---

## 4. プロジェクトドキュメントの保存確認

プロジェクトドキュメント保存ルールに従い、本トピックに関するドキュメントは以下の場所に保存されています：
- [`task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_restructuring_and_core_figures/task.md)
- [`implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_restructuring_and_core_figures/implementation_plan.md)
- [`walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/paper_restructuring_and_core_figures/walkthrough.md)
