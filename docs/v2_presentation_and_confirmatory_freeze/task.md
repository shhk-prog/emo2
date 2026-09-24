# タスクリスト: V2 Presentation および Confirmatory 最終凍結 (Freeze)

## 背景
成果物監査において、Behavioral、V1、V3、paper-summary infrastructure は freeze 可と判定された。
残る修正は V2 の generator 側（`scripts/summarize_v2_reorganization.py` および `v2/scripts/build_paper_summary.py`、`v2/primary/run_confirmatory_analysis.py`）に残っている整合性問題 7 点に絞られている。

## タスク項目

- [x] **1. `v2_h1_h2_reorganization.tex` の全`---`解消とH1a構成整理**
  - `table_v2_confirmatory.csv` の metric 名（`Reader Procrustes Distortion`, `Self Procrustes Distortion` 等）と `scripts/summarize_v2_reorganization.py` の検索キーを完全一致（exact match）させる。
  - H1a confirmatory table からは RSA を除外し、`Reader Procrustes Distortion` と `Self Procrustes Distortion` の2行のみとする（RSAはdescriptive geometry tableに配置）。
- [x] **2. H1b の Self 結果を canonical confirmatory table に追加**
  - `v2/scripts/build_paper_summary.py` において、`valence.reader.shift`, `valence.self.shift`, `arousal.reader.shift`, `arousal.self.shift` の4行すべてを出力するように修正。
  - metric 名を `Valence Reader Peak Shift Delta d*`, `Valence Self Peak Shift Delta d*`, `Arousal Reader Peak Shift Delta d*`, `Arousal Self Peak Shift Delta d*` 等とし、TeX側と完全同期。
- [x] **3. V2 H3 LMM の Valence/Arousal 両軸表示と Note の定性化**
  - `generate_h3_lmm_table()` で `table_v2_3c_lmm.csv` から Valence と Arousal の両軸（各3 Primary interactions + Secondary main effect）を表示。
  - Note から古いハードコード数値（$\beta=0.181$ 等）を削除し、Primary interaction terms が有意水準を満たさなかった旨の定性文に修正。
- [x] **4. V2 confirmatory summary Note の結果整合**
  - `v2_confirmatory_summary.tex` の Note を、表の判定（H1a Supported, H1b/H2 Not Supported, H3 Not Supported, H4 Incomplete）と完全に整合させる。
- [x] **5. H4 の 4-family completeness 判定への厳格化と表示整理**
  - `v2/primary/run_confirmatory_analysis.py` で `expected_families = {"qwen", "llama", "gemma", "olmo"}` に対する追跡を行い、揃わない場合は `h4_status = "incomplete"`。
  - 4/4 未完時は `diff_self_minus_reader_mean = descriptive only`, `CI = not confirmatory (---)`, `Supported = Incomplete / Not evaluated` とする。
  - Note を「現時点ではGemma familyのみ実測値が利用可能で、事前登録された4ファミリー設計に基づくH4確証的統合は未完」に修正。
- [x] **6. `table_v2_1_geometry.csv` の metric 名の曖昧性解消**
  - Center of Mass である `mean_dist` を `procrustes_distortion_center` に改名し、誤解を招く `mean_procrustes_distortion` 列を削除。
- [x] **7. `v2_reorganization_summary.md` のハードコード全廃と動的生成化**
  - `generate_markdown_summary()` の固定値を全廃し、canonical DataFrame (`df_conf`, `df_lmm`) から動的にMarkdown表を生成する。
- [x] **8. 再生成パイプライン実行と全件検証**
  - `build_all_paper_summaries.py --strict`
  - `generate_paper_results_tables.py`
  - `pytest tests/test_paper_summary_invariants.py`
  - 各 `.tex` 表の差分と内容を確認し、freeze 判定を行う。
