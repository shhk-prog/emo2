# タスク計画: 投稿前最終修正 & 匿名化コードZIP作成

## 目的
ICLR 2027二重盲検（Double-blind）投稿基準に適合する、完全匿名化かつ再現可能なSupplementary Material用コードパッケージを作成し、残存するテスト不整合を解消する。

## タスクリスト
- [x] 1. stale test の修正 (`tests/test_paper_summary_invariants.py::test_invariant_1_primary_orthogonality`)
  - Gate NO_GO 設計に基づき、V3 RQ2+ が primary_results.csv ではなく secondary_results.csv に含まれることを検証するよう修正
  - pytest 全件実行で 100% グリーン（156 passed, 2 deselected, 0 failed）を確認
- [x] 2. paper-summary の provenance & 上流 artifact の整理
  - `paper_summary_manifest.json` が参照する上流 artifact 49件の存在状況を監査（49件すべて実在を確認、計407KB）
  - `scripts/build_all_paper_summaries.py --strict` の正常終了を確認
  - 必要な derived artifacts の同梱範囲を確定し、README に再生成手順と依存関係を明記
- [x] 3. 匿名化・除外対象の徹底精査
  - ユーザー個人パス (`/mnt/nas/home/hiromi/...` 等) の含まれるファイルを全走査し修正（コード・スクリプト内ゼロ件達成）
  - 古いLaTeX原稿・テンプレート (`iclr2027/iclr2027_conference2.tex` 等) をコードZIPから除外
  - 不要な scratch/、古いログ、中間ファイル、git履歴などの除外
- [x] 4. 匿名化コードZIPのパッケージングスクリプトと検証
  - Supplementary 用 ZIP 生成＆自己検証スクリプト [`scripts/package_submission_code.py`](scripts/package_submission_code.py) を作成
  - 生成ZIPの二重盲検匿名性、49件上流artifact、`--strict` 再現、`pytest` 全件パスの自動検証パイプラインを実装
- [x] 5. ドキュメント（implementation_plan.md, walkthrough.md）の整理と報告
