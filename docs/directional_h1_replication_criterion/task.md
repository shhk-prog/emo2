# Task: Directional Decodability--Causal Dissociation (H1) Replication Criterion の修正と再集計

## 概要
V3のH1（Decodability--Causal Dissociation）において、現在の定義（$\Delta d^* = d_C^* - d_D^*$, $\Delta \bar{d} = \bar{d}_C - \bar{d}_D$）に基づくと、Qwen Discoveryでの観測結果は $d_C^* < d_D^* \Rightarrow \Delta d^* < 0$ であり、また重心（COM）についても Valence（$\Delta \bar{d} < 0$）と Arousal（$\Delta \bar{d} > 0$）で符号が異なっている。
それにもかかわらず、従来のcross-family H1では一律に $CI_{\text{low}} > 0$ を要求しており、Qwen Discoveryで発見されたパターンのreplication criterionになっていなかった。

本タスクでは、H1を「Qwen Discoveryで観測された方向のspatial dissociationが他familyでも再現するか（Directional Decodability--Causal Dissociation）」と論理的に再定義し、以下の改修を実施する：
1. Qwen Discoveryの符号（Valence / Arousal × peak / COM）の明示的freeze
2. `v3/primary/run_confirmatory_replication.py` の判定ロジック修正（Qwen符号によるdirection alignment、rawとalignedの両方の出力保持）
3. `v3/scripts/build_paper_summary.py` および `scripts/summarize_v3_causal_utilization.py` の再集計・再生成
4. 論文 `iclr2027/iclr2027_conference2.tex` の Methods（H1）、Results、Discussion の整合性更新
5. 単体テストの追加・更新と検証

## タスクリスト

- [x] 1. 事前準備・設計と freeze アーティファクトの生成 <!-- id: 1 -->
  - [x] 1.1 Qwen Discovery（`v3_spatiotemporal_summary.json`）から Valence/Arousal の peak および center-of-mass の符号を取得 <!-- id: 1.1 -->
  - [x] 1.2 `frozen_confirmatory_sites.json` に `h1_replication_direction` を追加保存 <!-- id: 1.2 -->
  - [x] 1.3 `v3/primary/run_rq3_path_mediation.py` の canonical freeze 出力にも `h1_replication_direction` を含めるよう更新 <!-- id: 1.3 -->

- [x] 2. `run_confirmatory_replication.py` の H1 判定ロジック修正 <!-- id: 2 -->
  - [x] 2.1 `frozen_confirmatory_sites.json` から `h1_replication_direction` を読み込み <!-- id: 2.1 -->
  - [x] 2.2 H1 出力辞書に raw 指標（`delta_peak_raw`, `delta_com_raw`, `delta_d_peak_ci`, `delta_d_center_ci`）と aligned 指標（`qwen_sign_peak`, `qwen_sign_com`, `delta_peak_aligned`, `delta_com_aligned`, `aligned_peak_ci`, `aligned_com_ci`）および各 pass 判定を保存 <!-- id: 2.2 -->
  - [x] 2.3 simulation（dry-run）および real 双方のパスで aligned CI lower > 0 を判定基準に設定 <!-- id: 2.3 -->

- [x] 3. 集計スクリプト・サマリーテーブルの更新と再集計 <!-- id: 3 -->
  - [x] 3.1 `v3/scripts/build_paper_summary.py` における H1 判定・抽出を新 criterion（aligned direction）に対応 <!-- id: 3.1 -->
  - [x] 3.2 `scripts/summarize_v3_causal_utilization.py` の LaTeX テーブル出力（`v3_confirmatory_details.tex`, `v3_confirmatory_matrix.tex`）を新判定・新ヘッダーに対応 <!-- id: 3.2 -->
  - [x] 3.3 保存済みデータから H1 判定を再計算・再集計し、`v3_cross_model_replication_summary.json`、`table_v3_4_confirmatory.csv`、`table_v3_confirmatory_matrix.csv`、LaTeX テーブル群を再生成 <!-- id: 3.3 -->

- [x] 4. 論文 `iclr2027/iclr2027_conference2.tex` の更新 <!-- id: 4 -->
  - [x] 4.1 Methods: `Replication H1：Directional Decodability--Causal Dissociation` の記述確認・調整 <!-- id: 4.1 -->
  - [x] 4.2 Results & Discussion: 再集計結果（ValenceではLlama/OLMoで同方向のdissociationが再現されたが、Arousalでは再現されず、3 families全体で一貫したdissociationとはならなかったこと等）を正確に反映 <!-- id: 4.2 -->

- [x] 5. テスト実行と検証 <!-- id: 5 -->
  - [x] 5.1 `tests/test_confirmatory_pipeline.py` に directional H1 alignment のユニットテストを追加 <!-- id: 5.1 -->
  - [x] 5.2 全テスト（pytest, ruff 等）を実行して検証 <!-- id: 5.2 -->
  - [x] 5.3 `walkthrough.md` の作成 <!-- id: 5.3 -->

- [x] 6. fail-fast バリデーションの厳格化と「事前登録」表記の適正化 <!-- id: 6 -->
  - [x] 6.1 `v3/primary/run_rq3_path_mediation.py`: 本番（`not args.dry_run`）での default フォールバック禁止、Qwen spatiotemporal summary 必須化、符号バリデーション <!-- id: 6.1 -->
  - [x] 6.2 `v3/primary/run_confirmatory_replication.py`: 本番（`not args.dry_run`）での `h1_replication_direction` 欠落時の default フォールバック禁止（KeyError 送出）と符号検証 <!-- id: 6.2 -->
  - [x] 6.3 `scripts/summarize_v3_causal_utilization.py`: LaTeX 表 caption/Note から「事前登録」を削除し、推奨の cross-family replication 表現に更新 <!-- id: 6.3 -->
  - [x] 6.4 `iclr2027/iclr2027_conference2.tex`: 本文中の「事前登録」「prespecified」表現の適正化と整合性確認 <!-- id: 6.4 -->
  - [x] 6.5 テーブル再生成（`summarize_v3_causal_utilization.py`, `build_paper_summary.py`）と全テスト（`PYTHONPATH=src:. pytest -q`）実行 <!-- id: 6.5 -->

- [ ] 7. 論文投稿前の残件解消（V2 H1a判定、TeX master正常化、用語・コードの厳格化） <!-- id: 7 -->
  - [ ] 7.1 V2 H1aの「Supported」判定を「Descriptive」へ修正（`v2/scripts/build_paper_summary.py`）＆ `table_v2_confirmatory.csv` 再生成 <!-- id: 7.1 -->
  - [ ] 7.2 V3 H1 Methods の $\widetilde{\Delta}$ 定義の数式・文章の完全復元・整頓 <!-- id: 7.2 -->
  - [ ] 7.3 `run_confirmatory_replication.py` の `run_real_model_confirmatory` における silent fallback 削除 <!-- id: 7.3 -->
  - [ ] 7.4 `v3/scripts/build_paper_summary.py` の silent fallback 削除（strict化） <!-- id: 7.4 -->
  - [ ] 7.5 `scripts/summarize_v3_causal_utilization.py` の V3 Matrix 表における confirmatory 語彙および Note の修正＆表再生成 <!-- id: 7.5 -->
  - [ ] 7.6 V3 Methods 内の旧語（confirmatory model/family等）の置換 <!-- id: 7.6 -->
  - [ ] 7.7 LaTeX 原稿の赤字メモ（`\color{red}` 等）全削除 <!-- id: 7.7 -->
  - [ ] 7.8 Qwen 2.5 の BibTeX citation 修正（`team2025qwen3` → 正式引用キー） <!-- id: 7.8 -->
  - [ ] 7.9 TeX master の構文エラー修正と ICLR サンプル重複の解消 <!-- id: 7.9 -->

