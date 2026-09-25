# タスクリスト: LaTeX考察・接続の徹底検証と重複修正（第2パス）

## 目的
`/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex` において、最新の実験結果（特に4ファミリー完全揃いとなったV2最新データ）と考察・接続の完全一致を検証し、AIコピペ残骸の完全除去、重複した議論のスリム化、過剰な別行立て数式の統合を行う。

## タスク項目
- [x] **1. 最新実験結果との整合性確認**
  - V2 H3 LMM: 全面的棄却ではなく、ValenceのAlignment $\times$ Task（$q=0.044$）およびAlignment主効果（$q=0.027$）が部分支持され、層深度再配置（Alignment $\times$ Depth等）は支持されなかった最新結果と完全整合。
  - V2 H4 Recovery: 4ファミリー実測値（Qwen, Llama, Gemma, OLMo）および介入方向（Instruct表現をBase表現で置換・整列）の定義と完全整合。
  - V3 Gate & Confirmatory: Gate NO_GO、H1-H3未達、H4 Temporal Contrastの有意シグナルと完全整合。
- [x] **2. ゴミテキスト・コピペ残骸の完全除去**
  - `:chatgpt-content-reference{index="3"}` (Line 8471) および `:chatgpt-content-reference{index="4"}` (Line 8555) を完全削除。
  - 全角カンマ「，」や重複コメントヘッダーの修正。
- [x] **3. 無駄な重複・寸断された数式レイアウトの修正**
  - 1行の数値や単語（`\[ 0.189 \]`, `\[ layer \times stage \]` 等）を別行立て数式ブロックにしている箇所をインライン・数式環境へ統合。
  - 「全体の考察」において各ステージ考察の単純な丸ごと再掲となっている部分を、総合考察（Synthesis）として引き締め。
- [x] **4. 検証と成果物レポート作成**
  - LaTeX構文およびセクション順序の完全性確認。
  - `walkthrough.md` の更新。
