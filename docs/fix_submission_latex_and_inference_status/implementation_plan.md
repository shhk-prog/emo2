# 実装計画: 論文提出用LaTeXおよび推論ステータスの最終修正

## 1. 目的
ICLR 2027投稿に向けて、論文TeX masterのコンパイルエラー（ICLR公式サンプル本文の残存、未閉じグループ）を解消し、State Induction Gate = NO_GO に伴う推論ステータス（exploratory / non-primary / descriptive）をコードおよび本文・表全体で整合させ、残存する古い言及や赤字修正メモ・不適切な引用等を完全にクリーンアップする。

## 2. 変更対象ファイルと作業内容

### (1) TeX master (`iclr2027/iclr2027_conference2.tex` または該当マスターTeX)
- sample本文（`Formatting Instructions for ICLR 2027 Conference Submissions`、sample authors、sample abstract、Sets and Graphsの未閉じグループ `\bgroup`、およびReferences後のsample Appendixなど）を完全に削除。
- 正しい投稿用構造（`\title{...} \maketitle \begin{abstract} ... \end{abstract} ... 実際のMain本文 \bibliography{...} \bibliographystyle{...} \appendix ...`）に整える。
- ユーザーが開いているのは `iclr2027/iclr2027_conference2.tex` なので、`iclr2027/iclr2027_conference.tex` との関係を確認した上で修正。
- コンパイル（`pdflatex`）がエラーなく通ることを確認。

### (2) `v3/scripts/build_paper_summary.py`
- Gate読み込み直後に `post_gate_primary = bool(pipeline_continues)` および `post_gate_role = ("confirmatory" if pipeline_continues else "exploratory")` を定義。
- RQ2: `is_primary=post_gate_primary, analysis_role="discovery"`
- RQ3: `is_primary=post_gate_primary, analysis_role=post_gate_role`
- Cross-family H1-H4: `is_primary=post_gate_primary, analysis_role=post_gate_role`
- H1 fallbackのハードコード辞書を削除し、`frozen_confirmatory_sites.json` が存在しない場合や `h1_replication_direction` がない場合は明確にエラーを上げるように修正。

### (3) `scripts/summarize_v2_reorganization.py`
- H1a判定を `l_ci > 0 or u_ci < 0` ではなく `supp_str = "Descriptive"` に修正。
- Noteの文言を「H1aはcross-family descriptive summaryとして扱う。Base--Instruct間にはReader / Self双方でrepresentation-geometric disparityが観測されたが、Procrustes distortionのCIが0を除外すること自体をnull-hypothesis testとは解釈しない。」に変更。
- 「事前登録仮説」「事前登録判定」等の表現を「V2 Cross-family Hypothesis Summary」「Key Analysis Metric」「Criterion / Interpretation」等に変更。

### (4) `scripts/build_all_paper_summaries.py`
- V2の核心的知見の文言を最新結果に合わせて修正。
- V3部分を「Table V3-3: Mediated Attenuation」「Table V3-4: Cross-family Replication」とし、核心文を「State Induction GateはValence / Arousal双方でNO_GOとなった。したがってRQ2以降はoverride下のexploratory / descriptive analysisとして扱う。Qwen Discoveryで選択されたsite / stage / directionはreplication-family outcomesを見る前にfreezeされた。」に変更。

### (5) LaTeX本文の記述修正 (`iclr2027/iclr2027_conference2.tex` / `.tex`)
- V3 MethodsのPrimary evidence表のキャプションにGate NO_GO条件および実結果でのexploratory/descriptive解釈を明記。held-out confirmationをheld-out cross-family replicationへ。
- 赤字修正メモ（`\color{red} 9/25 ...` と対応する `\color{black}`）の6箇所を削除（本文内容は維持）。
- Qwen 2.5のcitationを `team2025qwen3` から `yang2024qwen25` (Qwen2.5 Technical Report, arXiv:2412.15115) へ修正（BibTeXおよび本文引用箇所2箇所）。
- V2考察の見出しを「Base--Instruct間では情報消失よりも表現幾何の差が観測された」へ緩和。

### (6) `iclr2027/tables/v3_confirmatory_matrix.tex` および `scripts/summarize_v3_causal_utilization.py`
- NoteでH1（Qwen Discovery終了後、replication-family outcomesを評価する前にfreezeしたdirectional criterion）とH2--H4（prespecified threshold / CI criterion）を区別するように修正。

### (7) 再生成および検証
- 各修正スクリプトを実行し、LaTeXテーブル・サマリーを再生成。
- テストスイートの実行確認。
- `pdflatex` によるコンパイル確認。
