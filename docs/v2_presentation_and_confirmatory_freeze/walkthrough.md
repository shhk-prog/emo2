# 修正確認書 (Walkthrough): V2 Presentation および Confirmatory 最終凍結 (Freeze)

## 1. 実施概要
ユーザーからの指摘に基づき、成果物監査で残存していた **V2 の generator 側不整合 7 点** をすべて改修し、パイプラインの再実行とテストを完了しました。

---

## 2. 7点の修正内容と検証結果

| # | 指摘項目 | 実施内容 | 検証結果 |
|---|---|---|---|
| 1 | `v2_h1_h2_reorganization.tex` の全`---`解消とH1a構成整理 | `table_v2_confirmatory.csv` の metric 名（`Reader Procrustes Distortion`, `Self Procrustes Distortion`）と `summarize_v2_reorganization.py` を exact match に修正。H1a から RSA を除外し、歪み度合い（Distortion）の2行のみに整理。 | 全行に実測値とブートストラップ95%信頼区間が反映され、`---` は 0 件。<br>Reader: 1.455 [0.639, 2.813]<br>Self: 2.872 [0.660, 6.995] |
| 2 | H1b の Self 結果を canonical confirmatory table に追加 | `v2/scripts/build_paper_summary.py` で `valence.reader.shift`, `valence.self.shift`, `arousal.reader.shift`, `arousal.self.shift` の4行すべてを出力。名前を `Valence Reader Peak Shift Delta d*` 等とし、TeX側と完全同期。 | 4行すべてが出力され、TeX側でも Reader/Self の両方が描画された。 |
| 3 | V2 H3 LMM の両軸表示と Note 定性化 | `generate_h3_lmm_table()` で `table_v2_3c_lmm.csv` から Valence と Arousal の両軸（各3 Primary terms + Secondary main effect）を表示。Note から古いハードコード数値（$\beta=0.181$ 等）を削除し、定性的な正確な記述に統一。 | 両軸（Valence/Arousal）が表示され、Note も指定の定性文に改修完了。 |
| 4 | V2 confirmatory summary Note の結果整合 | 表の判定（H1a Supported, H1b/H2 Not Supported, H3 Not Supported, H4 Incomplete）と矛盾していた旧 Note を、正確な事前定義判定基準と結果に改修。 | Note が表の全仮説判定結果と完全に整合。 |
| 5 | H4 の 4-family completeness 判定厳格化 | `run_confirmatory_analysis.py` で `expected_families = {"qwen", "llama", "gemma", "olmo"}` に対する厳格な追跡を行い、未完の場合は `h4_status = "incomplete"`, `diff_bootstrap_ci_95 = None`。表上も `Incomplete / Not evaluated`、CI は `---` と明記。Note の誤記も修正。 | 推論過剰が排除され、descriptive only / Incomplete と明記。 |
| 6 | `table_v2_1_geometry.csv` の metric 名曖昧性解消 | Center of Mass（0.4〜0.6）である列名を `procrustes_distortion_center` に改名し、誤解を招く `mean_procrustes_distortion` 列を削除。 | distortion magnitude（1.455）と distortion center（0.58）の混同を完全に防止。 |
| 7 | `v2_reorganization_summary.md` の動的生成化 | `generate_markdown_summary()` 内の固定値（0.353, 0.425 等）を全廃し、`df_conf` および `df_lmm` から動的に Markdown 表を生成。 | Markdown が canonical CSV の最新実測値と 100% 同期。 |

---

## 3. 生成された表の確認

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
\multirow{2}{*}{H1a: Geometric Distortion} & Reader Procrustes Distortion         & 1.455    & [0.639, 2.813]     & Matched-Plain \\
                                 & Self Procrustes Distortion           & 2.872    & [0.660, 6.995]     & Matched-Plain \\
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
\textbf{Note:} Primary confirmatory inferenceは Alignment $\times$ Depth、Alignment $\times$ Task、Alignment $\times$ Task $\times$ Depth のprespecified interaction termsに基づく。ValenceおよびArousalのいずれにおいても、これらのPrimary interactionはFDR補正後の有意水準を満たさなかった。
\end{minipage}
\end{table}
```

---

## 4. 検証結果
- `tests/test_paper_summary_invariants.py`: **8/8 pass**
- 全体テスト（`pytest tests/`）: **154 passed, 0 failed**
- 表生成パイプライン: **差分ゼロで安定再生成可能**
- 不整合・ハードコード・空表はすべて解消。

これで結果フォルダの freeze 条件が完全に整いました。
