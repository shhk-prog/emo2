# Walkthrough: V2包括サマリーMarkdown (v2_reorganization_summary.md) の完全同期と全結果拡充

## 1. 修正の概要
1. `/mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v2_reorganization_summary.md` において、以前は「Confirmatory Hypotheses Testing Summary」および「Causal Relocation LMM (H3)」の2つの表しか収録されておらず、V2 Stageの全分析結果が欠落していました。
2. さらに、「## 2. Causal Peak Relocation (H3a)」および「## 3. Causal Specificity Controls (H3b)」の表において、全行が `---`（未評価・欠損）となっていました。
3. 原因を調査したところ、[`v2/scripts/build_paper_summary.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/scripts/build_paper_summary.py) が [`v2_causal_dissociation_summary.json`](file:///mnt/nas/home/hiromi/src/emo2/v2/results/derived/v2_causal_dissociation_summary.json) の `per_family[fam]["results"]["causal_maps"]` を参照せず直下の `causal_maps` を見ていたため、空辞書とみなされ NaN で初期化されていたことが判明しました。
4. 本対応により、データ集約パイプラインを修正し、V2 Stageの全6テーブルに正確な実測値を100%完全に反映させました。

## 2. 変更内容の詳細

### 2.1 データ取得パイプラインのバグ修正 ([`v2/scripts/build_paper_summary.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/scripts/build_paper_summary.py))
- `cd_data`（`v2_causal_dissociation_summary.json`）のパース処理において、`per_family[fam]` の下にある `"results"` 階層から `causal_maps` および `relative_depths` を取得するように修正：
  ```python
  results_entry = fam_entry.get("results", fam_entry) if isinstance(fam_entry, dict) else {}
  cmaps = results_entry.get("causal_maps", fam_entry.get("causal_maps", {}))
  rel_depths = results_entry.get("relative_depths", fam_entry.get("relative_depths", [0.0, 0.33, 0.67, 1.0]))
  ```
- これにより、全4ファミリー（Qwen 2.5 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B）の因果介入・統制測定値が `table_v2_3a_causal_relocation.csv` および `table_v2_3b_causal_controls.csv` に完全に出力されるようになりました。
- `build_all_paper_summaries.py` の Primary レコード数も 323 行から 435 行へ正常に拡充されました。

### 2.2 サマリー生成スクリプトの改修 ([`scripts/summarize_v2_reorganization.py`](file:///mnt/nas/home/hiromi/src/emo2/scripts/summarize_v2_reorganization.py))
- `generate_markdown_summary(df_conf, df_reloc, df_ctrl, df_lmm, df_recov)`:
  全5つの Canonical DataFrame をすべて渡して以下の6セクション構成で Markdown を動的生成するように実装：
  1. **`## 1. Representation Geometry and Sharing Reorganization (H1--H2)`**
     - H1a: 幾何学的歪み（Reader / Self Procrustes Distortion）
     - H1b: デコードピーク深度変位（$\Delta d^*$：Valence/Arousal × Reader/Self）
     - H2: タスク間表現共有度変位（$\Delta\text{Sharing}$：Valence/Arousal）
     - Note
  2. **`## 2. Causal Peak Relocation (H3a)`**
     - 4 Family（GEMMA, LLAMA, OLMO, QWEN）× 2 Task（Reader, Self）× 2 Axis（Arousal, Valence）の Base Peak、Instruct Peak、ピーク変位 $\Delta d_C^*$、重心変位 $\Delta d_{\text{center}}$ の**全実測値を完全反映**
     - Note（記述的集約値であること、および仮説検証は sample-level LMM に基づく旨を明記）
  3. **`## 3. Causal Specificity Controls (H3b)`**
     - 全32条件の生の因果効果 $C_{\text{raw}}$、ランダム統制 $C_{\text{rand}}$、直交統制 $C_{\text{perp}}$、正味因果効果 $C_{\text{net,rand}}$、およびゼロ切除効果の**全実測値を完全反映**
     - Note
  4. **`## 4. Causal Relocation Mixed-Effects Model (H3c LMM)`**
     - **4.1 Primary Interventions and Post-training Effects**: LaTeX版 Table 4 に対応する Valence および Arousal の事前登録交互作用項・事後学習主効果
     - Note（Valence の Alignment $\times$ Task $q = 0.044$ の部分的有意性と深度再配置棄却の注記）
     - **4.2 Full LMM Parameter Estimates**: 全24項目の固定効果・ランダム効果分散の詳細表
  5. **`## 5. Output Distribution Recovery (H4)`**
     - 4 Family × 2 Task の Matched AUC、$\Delta\text{EMD AUC}$、Max Recovery、Best Depth ($d^*$)、Native AUC、Aligned AUC
     - Note
  6. **`## 6. Pre-registered Hypotheses Testing Summary (H1--H4)`**
     - H1〜H4の全事前登録仮説の推定値、95% CI、FDR $q$ / $p$、Supported? 判定結果
     - Note

## 3. 生成されたファイル内容の確認

### Section 2: Causal Peak Relocation (H3a) 実測値サンプル
| Family | Task | Axis | Base Peak ($d^*$) | Instruct Peak ($d^*$) | $\Delta d_C^*$ | $\Delta d_{\text{center}}$ |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| GEMMA | Reader | Arousal | 0.760 | 0.560 | $-$0.200 | $-$0.106 |
| GEMMA | Reader | Valence | 0.520 | 0.080 | $-$0.440 | 0.098 |
| LLAMA | Self | Arousal | 0.067 | 0.800 | 0.733 | 0.254 |
| LLAMA | Self | Valence | 0.467 | 0.933 | 0.467 | 0.341 |
| OLMO | Reader | Valence | 0.600 | 0.267 | $-$0.333 | $-$0.170 |
| QWEN | Self | Arousal | 0.296 | 0.852 | 0.556 | 0.027 |
*(全16行に実測値が正しく反映)*

### Section 3: Causal Specificity Controls (H3b) 実測値サンプル
| Family | Task | Condition | Axis | $C_{\text{raw}}$ | $C_{\text{rand}}$ | $C_{\text{perp}}$ | $C_{\text{net,rand}}$ (Primary) | Zero-Ablation |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| OLMO | Reader | Base-Plain | Valence | 0.002 | 0.002 | 0.002 | $-$0.000 | 0.016 |
| OLMO | Reader | Matched-Plain | Valence | 0.004 | 0.004 | 0.004 | $-$0.000 | 0.053 |
| LLAMA | Reader | Base-Plain | Valence | 0.003 | 0.003 | 0.003 | 0.000 | 0.033 |
| GEMMA | Self | Matched-Plain | Valence | 0.006 | 0.006 | 0.006 | 0.000 | 0.059 |
| QWEN | Reader | Matched-Plain | Valence | 0.007 | 0.007 | 0.007 | $-$0.000 | 0.122 |
*(全32条件に実測値が正しく反映)*

## 4. 検証結果
1. **データ集約オーケストレーター**:
   ```bash
   .venv/bin/python scripts/build_all_paper_summaries.py
   ```
   終了コード0で完了（Primary Results: 435 rows, Tables: 25 files）。
2. **テーブル生成**:
   ```bash
   .venv/bin/python scripts/summarize_v2_reorganization.py
   ```
   終了コード0で完了し、MarkdownおよびLaTeXテーブルがすべて最新実測値と同期。
3. **自動テスト**:
   ```bash
   .venv/bin/python -m pytest -q tests/
   ```
   156 passed, 2 deselected, 5 warnings（全テスト通過）。
