# paper_v1.md の内容を日本語のまま意味や単語を変えずに iclr2027_conference2.tex に反映する実装計画

## 概要
`/mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md`（全4,797行）の全セクション・全文・全表・全数式・全考察を、**「意味や単語は一切変えずに」**、日本語のまま `/mnt/nas/home/hiromi/src/emo/iclr2027/iclr2027_conference2.tex` へ LaTeX 形式として正確に移植・変換します。

---

## ユーザー確認事項 (User Review Required)
- **コンパイル環境への配慮**:
  ICLRのスタイルファイル (`iclr2027_conference.sty`) は標準で英文向け（pdfLaTeX想定）ですが、日本語テキストを扱えるようにプリアンブルで日本語対応設定（LuaLaTeX向けの `luatexja` や、pdfLaTeX/XeLaTeX/upLaTeXいずれにも耐えうる汎用的な記述）を安全に追加します。
- **文言・意味の非破壊原則**:
  ユーザー様のご指示通り、文章の要約や自己流の言い換えは一切行わず、Markdownの記法（見出し、箇条書き、太字、表、数式、引用、コード等）を LaTeX 記法（`\section`、`\begin{itemize}`、`\textbf`、`\begin{tabular}`、`align` 等）へ機械的かつ厳密に変換します。

---

## 変換・実装ステップ

1. **プリアンブルおよび文書構造の準備 (`iclr2027_conference2.tex`)**:
   - `iclr2027_conference.sty` と日本語文字を共存させるプリアンブル（LuaLaTeX / XeLaTeX / pdfLaTeX-CJK 等を想定した設定、表組み用の `booktabs`, `tabularx`, `longtable`, `array`, `amsmath`, `amssymb` などのパッケージインポート）を整備。
   - タイトル・アブストラクト・著者の枠組みを適切に設定。

2. **`paper_v1.md` の各章を LaTeX 化**:
   - **第1章**: 序論・研究背景 (Introduction)（表1、参考文献リスト等含む）
   - **第2章**: 関連研究 (Related Work)
   - **第3章**: EmoBank 実験概要
   - **第4章**: EmoBank 3-Way VAD Evaluation Report（各モデル別詳細表、3×3マトリクス、横断分析）
   - **第5章**: EmoBank 実験 総合考察（General Discussion）
   - **第6章**: AIPsy-Affect 4-Split 実験概要
   - **第7章**: AIPsy-Affect 4-Split Comprehensive Evaluation Report
   - **第8章**: AIPsy-Affect 4-Split 実験 総合考察（General Discussion）
   - **第9章**: V1の概要：実験フレームワークと実装プロトコル
   - **第10章**: Phase Aの結果・考察
   - **第11章**: Phase Bの結果（E5監査レポート）およびPhase Bの考察
   - **第12章**: Phase Cの結果・考察

3. **LaTeX 特殊文字・記法の厳密なエスケープ**:
   - アンダースコア `_`、パーセント `%`、アンパサンド `&`、ハッシュ `#`、ドル記号 `$`、波括弧 `{}` 等のエスケープ漏れを防止。
   - Markdown 表（パイプ `|` 区切り）を `table` / `tabular` / `tabularx` または `longtable` へ変換。

4. **検証**:
   - TeX ファイル内の構文チェック、括弧の対応、特殊文字エスケープの確認。
   - 変更内容を `docs/paper_v1_to_tex/` に記録。

---

## 変更対象ファイル
- `iclr2027/iclr2027_conference2.tex` (全面更新)
- `docs/paper_v1_to_tex/task.md` (新規)
- `docs/paper_v1_to_tex/implementation_plan.md` (新規)
- `docs/paper_v1_to_tex/walkthrough.md` (完了後作成)
