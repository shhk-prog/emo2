# タスクリスト: 5つのレポートスクリプトへのTeX出力機能追加とiclr2027_conference2.texへの反映

- [x] 準備フェーズ <!-- id: 0 -->
  - [x] 対象5スクリプトと出力ファイルの特定 <!-- id: 1 -->
  - [x] 記録ディレクトリ `docs/export_reports_to_tex/` の整備 <!-- id: 2 -->
- [x] 実装フェーズ <!-- id: 3 -->
  - [x] 共通変換モジュール `v1/scripts/md_to_tex.py` の作成 <!-- id: 4 -->
  - [x] `v1/scripts/summarize_3way_vad.py` の修正（TeX出力追加） <!-- id: 5 -->
  - [x] `v1/scripts/summarize_aipsy_4split.py` の修正（TeX出力追加） <!-- id: 6 -->
  - [x] `v1/scripts/generate_phase_a_report.py` の修正（TeX出力追加） <!-- id: 7 -->
  - [x] `v1/scripts/generate_phase_b_report.py` の修正（TeX出力追加） <!-- id: 8 -->
  - [x] `v1/scripts/generate_phase_c_report.py` の修正（TeX出力追加） <!-- id: 9 -->
- [x] 実行・検証フェーズ <!-- id: 10 -->
  - [x] 5つのスクリプトを実行し、各ディレクトリに `.tex` レポートが生成されることを確認 <!-- id: 11 -->
  - [x] `iclr2027/iclr2027_conference2.tex` の空欄結果セクションへの反映 <!-- id: 12 -->
  - [x] `docs/export_reports_to_tex/walkthrough.md` の作成 <!-- id: 13 -->
