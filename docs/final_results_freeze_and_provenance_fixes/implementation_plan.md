# 実装計画: 結果フォルダの最終凍結・確証的判定およびプロベナンス修正

## 1. 背景と目的
ICLR 2027論文に向けた結果フォルダの監査において、BehavioralおよびV1はすでに整合的であることが確認された。
本計画では、残るV2（3点：H3確証判定、H4確証指標、RQ3/4ファミリー網羅性）、V3（1点：H3 $M_{\text{net}}$ 判定ロジック）、Provenance（1点：`results/derived/paper_summary/` のzip同梱）、および細部体裁（H2点推定値等）の計8項目を完全に修正し、結果フォルダを確定・凍結（freeze）する。

---

## 2. 修正対象ファイルと変更計画

### 2.1 V2確証的判定 (H3 & H4)
- **対象**:
  - `v2/scripts/build_paper_summary.py`
  - `scripts/summarize_v2_reorganization.py`
  - `iclr2027/tables/v2_confirmatory_summary.tex`
- **内容**:
  1. H3 Primary 項を事前登録仕様に基づき `Alignment × Depth`, `Alignment × Task`, `Alignment × Task × Depth` の3項目とし、FDR $q$ 値（0.878, 0.878, 0.961）を明記の上、判定を `Not Supported` とする。
  2. 主効果 `Alignment main effect` は Secondary / descriptive 行へ分離。
  3. H4 Primary 項を `Self--Reader Matched-Plain AUC Difference` とし、ブートストラップ95%信頼区間 `[-0.015, 0.078]` を提示。2/4ファミリーしか実測がないため `Not Supported (Incomp.)` とする。

### 2.2 V2 RQ3/RQ4 ファミリー網羅性 (4-Family Coverage)
- **対象**:
  - `results/derived/paper_summary/tables/table_v2_3a_causal_relocation.csv`
  - `results/derived/paper_summary/tables/table_v2_3b_causal_controls.csv`
  - `results/derived/paper_summary/tables/table_v2_4_recovery.csv`
  - `scripts/summarize_v2_reorganization.py`
  - `iclr2027/tables/v2_causal_relocation.tex`
  - `iclr2027/tables/v2_causal_controls.tex`
  - `iclr2027/tables/v2_distribution_recovery.tex`
- **内容**:
  1. OLMo 単一モデルではなく、4ファミリー（Gemma, Llama, OLMo, Qwen）を網羅する表構成とする。
  2. Gemma, OLMo, Qwen の実測因果指標を格納し、未集約の Llama は `---`（欠損）として明示。
  3. Recovery においても Gemma, OLMo の実測値を掲載し、Llama, Qwen が未実施であることを明記。

### 2.3 V3 H3判定ロジックの厳密化
- **対象**:
  - `v3/scripts/build_paper_summary.py`
  - `scripts/summarize_v3_causal_utilization.py`
- **内容**:
  - 媒介減衰効果 $M$ の判定において、$M > 0$ だけでなく、ランダム部分空間統制を差し引いた正味媒介減衰量 $M_{\text{net}} = M - M_{\text{rand}}$ の 95% 信頼区間下限が 0 を超えているか（$\text{CI}_{\text{low}}(M_{\text{net}}) > 0$）を Valence と Arousal の両軸で検証するようコードを修正。

### 2.4 H2 $\Delta\text{Sharing}$ の点推定値の補完
- **対象**:
  - `results/derived/paper_summary/tables/table_v2_confirmatory.csv`
  - `scripts/summarize_v2_reorganization.py`
  - `iclr2027/tables/v2_h1_h2_reorganization.tex`
  - `iclr2027/tables/v2_confirmatory_summary.tex`
- **内容**:
  - 点推定値（Valence: $-0.082$, Arousal: $-0.055$）を明記。

### 2.5 プロベナンス保証 (`results/derived/paper_summary/` の Git/Zip 追跡)
- **対象**:
  - `.gitignore`
- **内容**:
  - 親ディレクトリのワイルドカード除外 (`**/results/derived/**`) により遮断されていた `results/derived/paper_summary/` を否定ルール (`!results/derived/paper_summary/**`) で確実に Git 管理対象へ含める。
  - `test_paper_summary_invariants.py` の実行により、テーブルCSVと不変条件を検証。

---

## 3. 検証手順
1. `pytest -q tests/test_paper_summary_invariants.py` の実行（全パス確認）。
2. 全テストスイート `pytest -q` の実行（153 passed 確認）。
3. `python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables` の実行（全ステージの表生成が正常完了することを確認）。
4. `git status` で `results/derived/paper_summary/` 配下のCSVやマニフェストが Git 追跡対象に含まれていることを確認。
