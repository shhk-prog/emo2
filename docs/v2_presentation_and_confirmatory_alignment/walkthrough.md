# 作業完了確認書 (Walkthrough): V2 Presentation および Confirmatory 整合性の修正

## 1. 実施概要
ユーザーからの指示に基づき、成果物監査で残件となっていた **V2のpresentation / confirmatory整合性に関する10項目** をすべて修正しました。
これにより、Behavioral、V1、V3に加え、V2の確証的推論表および再生成パイプラインが完全に Methods・実験データと同期し、結果フォルダを凍結（freeze）して考察執筆へ進める状態を確立しました。

---

## 2. 修正項目と反映結果の検証

| # | 項目 | 修正内容 | 検証結果 |
|---|---|---|---|
| 1 | `v2_h1_h2_reorganization.tex` の全`---`解消 | `scripts/summarize_v2_reorganization.py` のキー検索を `hypothesis + metric` の exact match に修正 | Reader Distortion: 1.455 [0.639, 2.813]<br>Self Distortion: 2.872 [0.660, 6.995]<br>RSA Reader/Self: 0.505<br>全実数値が反映 |
| 2 | V2 H1b に Reader / Self 両行を追加 | Methods の $\Delta d_{D,T}^*$ ($T \in \{\text{Reader}, \text{Self}\}$) に合わせ、builder および LaTeX 生成側で Valence/Arousal $\times$ Reader/Self 計4行を出力 | Valence Reader: -0.119 [-0.219, -0.033]<br>Valence Self: 0.007 [-0.053, 0.100]<br>Arousal Reader: 0.200 [-0.027, 0.426]<br>Arousal Self: 0.017 [-0.200, 0.250] |
| 3 | V2 confirmatory summary Note の修正 | 表自身の判定（H1a Supported, H1b/H2 Not Supported）と矛盾していた旧 Note を修正 | 正確な事前登録判定基準と判定結果に整合 |
| 4 | `v2_h3_causal_lmm.tex` Note のハードコード削除 | 別世代の古い数値（$\beta=0.181, p<0.001$ 等）を削除し、定性的な正確な記述に統一 | FDR補正後の有意水準に関する堅牢な定性記述に変更 |
| 5 | V2 H3 table の両軸表示 | Valence のみ表示されていた表を、Valence および Arousal の両軸（各3 primary interaction terms + secondary main effect）を表示する構成に拡張 | Valence と Arousal の全交互作用項（$q = 0.122 \sim 0.796$）を併記 |
| 6 | H4 の 4-family completeness 判定への厳格化 | `run_confirmatory_analysis.py` で事前定義された4ファミリー（qwen, llama, gemma, olmo）が揃っていない場合は `h4_status = "INCOMPLETE"`（Not evaluated / descriptive only）とするよう厳格化 | 判定: `Incomplete / Not evaluated`<br>CI: `---`<br>推論過剰を防止 |
| 7 | H4 Note の誤記修正 | 「GemmaおよびOLMoの実測値」という誤記を「現時点ではGemma familyのみ実測値が利用可能」に修正 | 実験実態（Gemmaのみ完全実測）と完全に整合 |
| 8 | V2 geometry CSV の列名分離 | Center-of-Mass が入っていた `mean_procrustes_distortion` を改名し、`procrustes_distortion_center` と真の `mean_procrustes_distortion`（layer-wise mean）を分離保存 | Center-of-mass（0.4〜0.6）と mean distortion（0.5〜9.0）の混同を解消 |
| 9 | V2 Markdown summary の動的生成化 | `v2_reorganization_summary.md` に直接書かれていた古いハードコード数値を全廃し、canonical CSVから動的生成 | CSVの正本数値がMarkdownにも完全反映 |
| 10 | 再生成パイプライン・テスト検証 | `build_all_paper_summaries.py --strict` および `generate_paper_results_tables.py` を再実行。不変条件テストおよび全体テストを実行 | `test_paper_summary_invariants.py`: 8/8 pass<br>全テスト: 154 passed, 0 failed |

---

## 3. 生成された主要 LaTeX 表の確認

### `iclr2027/tables/v2_h1_h2_reorganization.tex`
```latex
\begin{table}[htbp]
\centering
\small
\caption{V2 H1--H2（幾何再編と表現共有度の変位）：事後学習に伴う表現幾何学的歪み（Procrustes Distortion）、デコードピーク深度変位 $\Delta d^*$、およびタスク共有度変化 $\Delta\text{Sharing}$。4ファミリー統合ブートストラップ95\%信頼区間。}
\label{tab:v2_h1_h2_reorganization}
\begin{tabular}{ll ccc}
\toprule
\textbf{Hypothesis} & \textbf{Metric} & \textbf{Estimate} & \textbf{95\% CI} & \textbf{Condition} \\
\midrule
\multirow{4}{*}{H1a: Geometric Distortion} & Reader Procrustes Distortion         & 1.455    & [0.639, 2.813]     & Matched-Plain \\
                                 & Self Procrustes Distortion           & 2.872    & [0.660, 6.995]     & Matched-Plain \\
                                 & RSA Reader ($\rho_{\text{RSA}}$)     & 0.505    & [0.330, 0.673]     & Matched-Plain \\
                                 & RSA Self ($\rho_{\text{RSA}}$)       & 0.505    & [0.324, 0.698]     & Matched-Plain \\
\midrule
\multirow{4}{*}{H1b: Decodability Peak Shift} & Valence Reader $\Delta d^*$          & $-$0.119 & [$-$0.219, $-$0.033] & Matched-Plain \\
                                 & Valence Self $\Delta d^*$            & 0.007    & [$-$0.053, 0.100]  & Matched-Plain \\
                                 & Arousal Reader $\Delta d^*$          & 0.200    & [$-$0.027, 0.426]  & Matched-Plain \\
                                 & Arousal Self $\Delta d^*$            & 0.017    & [$-$0.200, 0.250]  & Matched-Plain \\
\midrule
\multirow{2}{*}{H2: Sharing Reorganization} & $\Delta\text{Sharing}$ (Valence)     & $-$0.553 & [$-$1.324, 0.076]  & Matched-Plain \\
                                 & $\Delta\text{Sharing}$ (Arousal)     & $-$0.278 & [$-$0.615, 0.049]  & Matched-Plain \\
\bottomrule
\end{tabular}
\vspace{1ex}
\begin{minipage}{\linewidth}
\footnotesize
\textbf{Note:} $\Delta d^* = d^*_{\text{instruct}} - d^*_{\text{base}} > 0$ は、事後学習によって表現デコードピークが後段側（deeper側）へシフトしたことを示す。また、$\Delta\text{Sharing} < 0$ はReaderとSelfの表現共有度合いが事後学習によってタスク分離方向に再編されたことを示す。
\end{minipage}
\end{table}
```

### `iclr2027/tables/v2_h3_causal_lmm.tex`
```latex
\begin{table}[htbp]
\centering
\small
\caption{V2 H3c（因果再配置の線形混合効果モデル LMM）：control-adjusted causal leverage $C_{\mathrm{net,rand}}$ を目的変数とし、family、alignment、task、relative depth、およびそのinteractionを評価したsample-level regression / mixed-effects analysis。ValenceおよびArousalの両軸におけるPrimary事前登録交互作用項および事後学習主効果。}
\label{tab:v2_h3_causal_lmm}
\begin{tabular}{ll cccc}
\toprule
\textbf{Axis} & \textbf{Predictor / Parameter} & \textbf{Estimate ($\beta$)} & \textbf{95\% CI} & $p$-value & \textbf{FDR $q$} \\
\midrule
\multirow{4}{*}{\textbf{Valence}} & $\text{Post-training} \times \text{Depth}$ (\textbf{Primary}) & 0.000147       & [$-$0.000050, 0.000343]    & 0.144      & 0.287      \\
                                 & $\text{Post-training} \times \text{Task}$ (\textbf{Primary}) & 0.000192       & [0.000030, 0.000354]       & 0.020      & 0.122      \\
                                 & $\text{Post-training} \times \text{Task} \times \text{Depth}$ (\textbf{Primary}) & $-$0.000213    & [$-$0.000491, 0.000065]    & 0.133      & 0.287      \\
                                 & Post-training ($\text{Instruct} = 1$) (\textbf{Secondary})   & $-$0.000114    & [$-$0.000229, 0.000000]    & 0.051      & ---        \\
\midrule
\multirow{4}{*}{\textbf{Arousal}} & $\text{Post-training} \times \text{Depth}$ (\textbf{Primary}) & $-$0.000008    & [$-$0.000061, 0.000044]    & 0.760      & 0.796      \\
                                 & $\text{Post-training} \times \text{Task}$ (\textbf{Primary}) & 0.000006       & [$-$0.000038, 0.000049]    & 0.796      & 0.796      \\
                                 & $\text{Post-training} \times \text{Task} \times \text{Depth}$ (\textbf{Primary}) & 0.000016       & [$-$0.000058, 0.000090]    & 0.676      & 0.796      \\
                                 & Post-training ($\text{Instruct} = 1$) (\textbf{Secondary})   & $-$0.000005    & [$-$0.000036, 0.000025]    & 0.740      & ---        \\
\bottomrule
\end{tabular}
\vspace{1ex}
\begin{minipage}{\linewidth}
\footnotesize
\textbf{Note:} Primary confirmatory inferenceは、Alignment $\times$ Depth、Alignment $\times$ Task、および Alignment $\times$ Task $\times$ Depth の事前登録された因果交互作用項に基づく。これらはいずれもBH-FDR多重比較補正後の有意水準を満たさなかった。
\end{minipage}
\end{table}
```

---

## 4. 結論
ご提示いただいた10件の指摘事項はすべて対応を完了しました。
- `tests/test_paper_summary_invariants.py`: 8/8 pass
- 全体テスト: 154 passed
- 表生成パイプライン: 正常終了、全表が canonical CSVから完全決定論的に再生成可能
- 結果フォルダ（`results/derived/paper_summary/` および `iclr2027/tables/`）の全整合性を確認済み

結果フォルダを凍結（freeze）し、論文の考察（Discussion / Conclusion）の執筆・更新へ進める準備が完了しました。
