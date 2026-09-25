# 実装計画: 投稿前最終修正 & 匿名化コードZIP作成

## 1. 概要
本計画は、ICLR 2027二重盲検要件を満たすクリーンなSupplementaryコードパッケージを作成し、論文および現在のGate設計（NO_GO時のV3 RQ2+非Primary化）と一致するテストの修正、およびpaper-summaryのprovenance整合性を確保することを目的とする。

## 2. 変更内容の詳細

### 2.1 Stale Test修正 (`tests/test_paper_summary_invariants.py`)
- **背景**:
  - State Induction GateがNO_GOとなったため、パイプライン設計に従い `post_gate_primary = bool(pipeline_continues)` (False) となり、RQ2以降はPrimaryではなくExploratory (Secondary) として記録されている。
  - これは論文記述と完全に一致する。
  - しかし `test_invariant_1_primary_orthogonality` はV3 RQ2 Discoveryが `primary_results.csv` に存在することを旧仕様としてアサートしているため失敗している。
- **修正内容**:
  - `table_v3_1_gate.csv` または `qc_summary.json` の `pipeline_continues` フラグを参照。
  - `pipeline_continues` がFalseの場合、V3 RQ2 Discoveryは `primary_results.csv` に存在**せず**、`secondary_results.csv` に存在することを検証する。
  - `pipeline_continues` がTrueの場合のみ `primary_results.csv` に存在することを検証する。
  - これによりGateの状態に応じた学術的不変条件テストとする。

### 2.2 Paper-Summary Provenance と Artifact の整理
- **現状**:
  - `results/derived/paper_summary/` 内の統合CSV・JSON・Markdownは生成済み。
  - 一方、`manifest` が参照する上流artifact 49件（各ステージのderived artifacts）の存在状況を監査。
  - `scripts/build_all_paper_summaries.py --strict` がどのartifact欠落で停止するかを特定。
- **対応方針**:
  - Supplementaryコードに最低限含めるべき derived artifacts（各ステージの集計結果）を確定。
  - もしZIP容量や設計上、生データ（raw）からの完全再生成ではなくderived成果物からの論文集約再現を主とする場合、必要なderived artifactsを同梱し、READMEに各ステージの実験スクリプト実行手順と論文サマリー構築手順（`--strict` 対応）を明記する。

### 2.3 匿名化と不要ファイル除外
- **除外・クリーンアップ対象**:
  - ユーザー個人パス (`/mnt/nas/home/hiromi/...` 等) を含むドキュメント、スクリプト、ログ
  - `iclr2027/iclr2027_conference2.tex` などの古いLaTeX原稿・テンプレート
  - `scratch/`、中間出力、不要な一時ファイル、古い実験ログ、`.git`
- **ZIPパッケージング自動化**:
  - `scripts/package_submission_code.py`（またはシェルスクリプト）を作成/整備し、除外ルールと匿名化チェックを自動化。
  - 生成されたZIPを展開したディレクトリに対して、`grep -rn "/mnt/nas" .` や `pytest` を実行し、完全性を自動検証する。

## 3. 検証計画
1. `pytest -v tests/test_paper_summary_invariants.py` の実行
2. 全体テスト `PYTHONPATH=src pytest` の実行（全パス確認）
3. `scripts/build_all_paper_summaries.py --strict` の実行確認
4. パッケージングスクリプトによるZIP生成と、解凍先での二重盲検パス・テスト検証
