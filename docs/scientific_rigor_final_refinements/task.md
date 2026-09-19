# タスクリスト: 科学的再現性・測定規約・スキーマの最終厳密化

- [x] 1. 実装計画の策定と承認 <!-- id: 1 -->
  - [x] 指摘された全13項目のコードベース詳細調査 <!-- id: 1.1 -->
  - [x] `task.md` および `implementation_plan.md` の作成 <!-- id: 1.2 -->
  - [x] ユーザー承認（Proceed）の受領 <!-- id: 1.3 -->

- [x] 2. 【P0】Behavioral 集計スクリプトの実行停止バグ修正 <!-- id: 2 -->
  - [x] `behavioral/analysis/summarize_behavioral_aipsy.py` の `cn_vals` 未定義 NameError を修正 <!-- id: 2.1 -->
  - [x] `tests/test_behavioral_specificity.py` を新規作成し、RQ3 集計の完走を検証 <!-- id: 2.2 -->

- [x] 3. 【P0/P1】V1 テストの API 不一致および戻り値スキーマ修正 <!-- id: 3 -->
  - [x] `v1/primary/run_phase_a.py` の `evaluate_cross_decoding_and_geometry` において、グループ不足時の戻り値スキーマを通常時と完全一致（NaN / "insufficient_groups" / True）に修正 <!-- id: 3.1 -->
  - [x] `tests/test_v1_token_and_probe_alignment.py` の import 関数名を `evaluate_cross_decoding_and_geometry` に修正し、高速テストをオールグリーン化 <!-- id: 3.2 -->

- [x] 4. 【P1】V1 E6 の 729 候補文字列形式統一 <!-- id: 4 -->
  - [x] `v1/primary/phase_c/run_e6_specialization.py` の独自スペース入り `build_vad_candidates` を完全削除し、`from affective_empathy_eval.likelihood import build_vad_candidates` に統一 <!-- id: 4.1 -->
  - [x] E6 候補と共通候補の完全一致アサーションをテストに追加 <!-- id: 4.2 -->

- [x] 5. 【P1】Manifest メタデータとデフォルト値の適正化 <!-- id: 5 -->
  - [x] `src/affective_empathy_eval/manifests.py` の `DEFAULT_INTERVENTION_VERSION` を `"none"` に変更 <!-- id: 5.1 -->
  - [x] `behavioral/primary/run_behavioral_emobank.py` および `run_behavioral_aipsy.py` で `candidate_space="VAD_729"`, `dataset_path=str(stim_path)`, `intervention_version="none"` を明示 <!-- id: 5.2 -->
  - [x] `v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`, `run_e6_specialization.py` で `candidate_space="VAD_729"` を明示 <!-- id: 5.3 -->

- [x] 6. 【P1】dtype の統一（特に V2 内部の一貫性） <!-- id: 6 -->
  - [x] `v2/primary/run_rq1_rq2_cross_decoding.py` のモデルロード dtype を `torch.bfloat16` に統一（RQ1〜RQ4 で完全一致） <!-- id: 6.1 -->

- [x] 7. 【P1】729 VAD vs 81 VA 感度分析の実装 <!-- id: 7 -->
  - [x] `scripts/run_candidate_space_sensitivity.py` を実装（同一刺激に対する 729 VAD と 81 VA の相関・符号一致・順序一貫性を算出） <!-- id: 7.1 -->
  - [x] 単体・dry-run テストによる完走確認 <!-- id: 7.2 -->

- [x] 8. 【P1】V2 H4 統計のペア／サンプルレベルへの強化 <!-- id: 8 -->
  - [x] `v2/primary/run_rq4_recovery_patching.py` で pair/sample レベルのレコードを保存 <!-- id: 8.1 -->
  - [x] `v2/primary/run_confirmatory_analysis.py` で 4-family bootstrap に加え、サンプルレベルの統計要約を生成 <!-- id: 8.2 -->

- [x] 9. 【P1】V3 frozen confirmatory sites の JSON artifact 化 <!-- id: 9 -->
  - [x] Discovery RQ2 終了時に `v3/results/derived/frozen_confirmatory_sites.json` を出力・保存 <!-- id: 9.1 -->
  - [x] `v3/primary/run_confirmatory_replication.py` で固定サイトファイルを読み込み、後続ファミリーの検証を実施 <!-- id: 9.2 -->

- [x] 10. 【P1】乱数シードの config 一元管理 <!-- id: 10 -->
  - [x] V3 の各スクリプトでハードコードされた seed を `config["seed"]` からの派生（`seed + 1`, `seed + 2`）に統一 <!-- id: 10.1 -->

- [x] 11. 【P1】論文アウトライン・各 Stage README・表記の統一 <!-- id: 11 -->
  - [x] `paper_outline.md` の V2 の説明を現行 production コード（Held-out cross-decoding, Procrustes, peak relocation, distribution recovery）に更新 <!-- id: 11.1 -->
  - [x] 各 Stage README（Behavioral, V1, V2, V3）の Section 番号（§4〜§7）および V1 タイトル（`in Base Models` 削除）を統一 <!-- id: 11.2 -->
  - [x] Root `README.md` の Markdown 表から `\multicolumn` を除去し、注記段落へ整形 <!-- id: 11.3 -->

- [x] 12. 【P1】pytest 設定の整理（高速・低速の分離） <!-- id: 12 -->
  - [x] `pyproject.toml` に `addopts = "-m 'not slow'"` を設定し、通常 CI の高速完走を保証 <!-- id: 12.1 -->

- [x] 13. 検証とテスト <!-- id: 13 -->
  - [x] `py_compile` による全スクリプト構文検証 <!-- id: 13.1 -->
  - [x] `pytest` による高速テストスイート（全 pass）の確認 <!-- id: 13.2 -->
  - [x] `docs/scientific_rigor_final_refinements/walkthrough.md` の作成 <!-- id: 13.3 -->
