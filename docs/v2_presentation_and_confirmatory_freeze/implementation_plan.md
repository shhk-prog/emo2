# 実装計画: V2 Presentation および Confirmatory 最終凍結 (Freeze)

## 1. 目的
成果物監査で特定された、V2 の generator 側（`scripts/summarize_v2_reorganization.py`, `v2/scripts/build_paper_summary.py`, `v2/primary/run_confirmatory_analysis.py`）に残存する 7 点の不整合を修正し、結果フォルダを完全に凍結（freeze）可能な状態にする。

## 2. 修正方針

### 1. `v2_h1_h2_reorganization.tex` の exact match 化と H1a 構成
- `table_v2_confirmatory.csv` の metric 名：
  - H1a: `Reader Procrustes Distortion`, `Self Procrustes Distortion`
  - H1b: `Valence Reader Peak Shift Delta d*`, `Valence Self Peak Shift Delta d*`, `Arousal Reader Peak Shift Delta d*`, `Arousal Self Peak Shift Delta d*`
  - H2: `Valence Delta Sharing`, `Arousal Delta Sharing`
- `scripts/summarize_v2_reorganization.py` の `generate_h1_h2_table()`:
  - exact match:
    ```python
    sub = df_conf[
        (df_conf["hypothesis"] == h_query)
        & (df_conf["metric"] == m_query)
    ]
    ```
  - H1a にはユーザー推奨通り `Reader Procrustes Distortion` と `Self Procrustes Distortion` の2行のみを配置（RSA は geometry descriptive table に置く）。

### 2. H1b の Self 結果追加
- `v2/scripts/build_paper_summary.py` の H1b 処理ブロック:
  ```python
  for mk, label in [
      ("valence.reader.shift", "Valence Reader Peak Shift Delta d*"),
      ("valence.self.shift", "Valence Self Peak Shift Delta d*"),
      ("arousal.reader.shift", "Arousal Reader Peak Shift Delta d*"),
      ("arousal.self.shift", "Arousal Self Peak Shift Delta d*"),
  ]:
  ```
  `primary_effects` から `mean_shift` と `bootstrap_ci_95` を取得し、正方向シフト基準（`ci[0] > 0.0`）で判定して `table_v2_confirmatory.csv` に出力。

### 3. V2 H3 LMM の両軸表示と Note 定性化
- `generate_h3_lmm_table()`:
  - 表ヘッダー: `Axis & Predictor / Parameter & Estimate ($\beta$) & 95\% CI & $p$-value & \textbf{FDR $q$}`
  - Valence:
    - $\text{Post-training} \times \text{Depth}$ (Primary)
    - $\text{Post-training} \times \text{Task}$ (Primary)
    - $\text{Post-training} \times \text{Task} \times \text{Depth}$ (Primary)
    - Post-training ($\text{Instruct} = 1$) (Secondary)
  - Arousal:
    - 同上4項目
  - Note: ハードコード数値を全廃し、Primary interaction terms が FDR 補正後に有意水準を満たさなかった旨の定性文にする。

### 4. V2 confirmatory summary Note の結果整合
- Note を以下に更新:
  ```latex
  \textbf{Note:}
  H1aではBase--Instruct間のgeometric distortionが確認された。
  一方、H1bのprespecified positive peak shiftおよび
  H2のReader--Self sharing reorganizationは、
  4-family bootstrap CIに基づく事前定義criterionを満たさなかった。
  H3のPrimary interaction termsもFDR補正後には支持されなかった。
  H4はprespecified 4-family resultsが揃っていないため、
  confirmatory conclusionを行わない。
  ```

### 5. H4 の 4-family completeness 判定厳格化
- `run_confirmatory_analysis.py`:
  - `expected_families = {"qwen", "llama", "gemma", "olmo"}`
  - 4/4 揃っていない場合は `h4_status = "incomplete"`, `diff_bootstrap_ci_95 = None`, `reorganization_supported = False`
- `build_paper_summary.py`:
  - `supported = "Incomplete / Not evaluated"`
  - `ci_low = np.nan`, `ci_high = np.nan`
- Note: 「現時点ではGemma familyのみ実測値が利用可能である。したがって、事前登録された4ファミリー設計に基づくH4確証的統合は未完（Incomplete）であり、confirmatory conclusionを行わない。」

### 6. `table_v2_1_geometry.csv` の metric 名
- `procrustes_distortion_center`: center of mass
- 誤解を招く `mean_procrustes_distortion` 列を削除。

### 7. `v2_reorganization_summary.md` の動的生成化
- `generate_markdown_summary()` の固定値を全廃し、`df_conf` および `df_lmm` から動的に行を組み立てて出力。

## 3. 検証手順
1. コード修正後、`python v2/primary/run_confirmatory_analysis.py` を実行。
2. `python scripts/build_all_paper_summaries.py --strict` で全サマリー再生成。
3. `python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables` で表再生成。
4. `pytest -q tests/test_paper_summary_invariants.py` の 8/8 pass を確認。
5. `git diff iclr2027/tables/` を確認し、全ての `---` 解消、両軸表示、Note 整合を確認。
