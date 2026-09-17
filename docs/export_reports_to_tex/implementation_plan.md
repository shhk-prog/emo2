# 5つのレポート生成スクリプトへの TeX 自動出力機能追加 実装計画

## 概要
以下の5つの主要レポート生成スクリプトを修正し、Markdown（`.md`）の出力に加えて、**「まったく同じ内容」が LaTeX（`.tex`）としても同時に自動出力**されるように改修します。

1. `/mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_detailed_report.md`
   $\rightarrow$ `3way_vad_detailed_report.tex`
   （対象スクリプト: `v1/scripts/summarize_3way_vad.py`）
2. `/mnt/nas/home/hiromi/src/emo/v1/results/aipsy_4split_eval/aipsy_4split_detailed_report.md`
   $\rightarrow$ `aipsy_4split_detailed_report.tex`
   （対象スクリプト: `v1/scripts/summarize_aipsy_4split.py`）
3. `/mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md`
   $\rightarrow$ `phase_a_comprehensive_report.tex`
   （対象スクリプト: `v1/scripts/generate_phase_a_report.py`）
4. `/mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_b/phase_b_comprehensive_report.md`
   $\rightarrow$ `phase_b_comprehensive_report.tex`
   （対象スクリプト: `v1/scripts/generate_phase_b_report.py`）
5. `/mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_c/phase_c_comprehensive_report.md`
   $\rightarrow$ `phase_c_comprehensive_report.tex`
   （対象スクリプト: `v1/scripts/generate_phase_c_report.py`）

さらに、生成された最新の `.tex` 内容を `/mnt/nas/home/hiromi/src/emo/iclr2027/iclr2027_conference2.tex` の該当箇所（結果セクション）にも反映・結合します。

---

## ユーザー確認事項 (User Review Required)
- **非破壊原則**:
  既存の Markdown 生成処理、集計ロジック、CSV 出力は一切変更せず、Markdown 生成直後に TeX 変換エンジンを通じて同一内容の `.tex` ファイルを書き出す処理を付加します。
- **TeX 構文整合性**:
  Markdown の複雑な多段表（テーブル）、数式、特殊文字（`_`, `%`, `&`, `#` 等）が LaTeX でコンパイルエラーを起こさないよう、厳密なエスケープと表環境（`tabular`）の整形を行います。

---

## 実装ステップ

1. **高精度 Markdown $\rightarrow$ TeX 変換共通モジュールの配備**:
   - `v1/scripts/md_to_tex.py` を作成。
   - 見出しレベルの TeX コマンド変換、Markdown テーブルの `tabular` 変換、数式ブロックの保護、インラインスタイル（`\textbf`, `\texttt`, `\textit`）変換、LaTeX 特殊文字の安全なエスケープを共通処理化。

2. **5つのスクリプトの修正**:
   - 各スクリプトのレポート保存処理に、対応する `.tex` ファイルの出力処理を追加。
   - オプション引数（`--out_tex`）もサポートし、デフォルトで `.md` と同一ディレクトリの同名 `.tex` へ保存。

3. **各スクリプトの実行と `.tex` ファイルの生成**:
   - 5つのスクリプトを実行し、最新データに基づく 5 つの `.tex` ファイルを出力。

4. **`iclr2027_conference2.tex` への結果セクションの反映**:
   - 空欄となっていた結果セクション（EmoBank 3-Way VAD、AIPsy 4-Split、Phase A、Phase B、Phase C）に、最新の TeX レポート内容を統合。

5. **検証と記録**:
   - 生成された `.tex` ファイルの内容確認。
   - `docs/export_reports_to_tex/` に `task.md`, `implementation_plan.md`, `walkthrough.md` を整備。
