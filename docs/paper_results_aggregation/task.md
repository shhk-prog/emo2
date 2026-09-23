# タスク：各Stage（Behavioral, V1, V2, V3）の論文結果集計コードの体系的実装

## 目的
論文構成仕様（§4 Behavioral, §5 V1, §6 V2, §7 V3）および確定した設計原則（確定最終版・完全合意版）に基づき、各ステージおよび全体統括の結果集計コードを作成する。
元実験スクリプトが確定した統計量をそのまま引き継ぎ、論文構造（表・図・長形式スキーマ）へ写像する純粋なプレゼンテーション層（read-only presentation layer）として構築する。

## ステータス / チェックリスト
- [x] 1. 実装計画書（`docs/paper_results_aggregation/implementation_plan.md`）の策定・完全合意版確定
  - V3 `response_end` テストの Gate 依存化（RQ2実行時のみ必須）
  - V2 `c_net_rand` 等の canonical metric における direct copy 原則徹底
  - 複数 artifact / 複数 key に対応した provenance manifest 構造
  - Unique-Key 衝突防止規約（`metric="lmm_beta::<term>"` 等による一意化）
- [x] 2. 共通19列スキーマ、複数artifact/key対応Provenance管理（`paper_summary_manifest.json`）、バリデータの作成
  - `src/affective_empathy_eval/paper_summary/schema.py`
  - `src/affective_empathy_eval/paper_summary/common.py`
- [x] 3. Behavioral 結果集計コード（Table B1〜B5, Fig B1〜B4, Model Summary（横並びのみ、スコア・ランクなし））の実装
  - `behavioral/analysis/build_paper_summary.py`
- [x] 4. V1 結果集計コード（Table V1-1〜V1-6, Fig V1-1〜V1-4, V1 Phase B nonfallback N併記）の実装
  - `v1/scripts/build_paper_summary.py`
- [x] 5. V2 結果集計コード（Table V2-1〜V2-4, C_net_rand direct copy, NaN保持, LMM term一意化）の実装
  - `v2/scripts/build_paper_summary.py`
- [x] 6. V3 結果集計コード（Gate文字列保持, Discovery vs Confirmatory直交分離, Gate依存response_end, Table V3-1〜V3-4）の実装
  - `v3/scripts/build_paper_summary.py`
- [x] 7. マスター集約オーケストレーターの実装
  - `scripts/build_all_paper_summaries.py`
  - `primary_results.csv`, `secondary_results.csv`, `paper_summary_manifest.json`, `qc_summary.json`, `results_summary.md`
- [x] 8. 論文上の条件付き不変条件テストの実装と検証
  - `tests/test_paper_summary_invariants.py`
  - 7つの条件付き不変条件（Primary直交性、NaN維持、Gate依存response_end存在、V1 Phase B nonfallback N、Gate判定ロジック、LMM term一意性・必須strata網羅、pipeline_continues依存のstrict検証）
- [x] 9. ウォークスルー文書（`docs/paper_results_aggregation/walkthrough.md`）の作成と完了報告
