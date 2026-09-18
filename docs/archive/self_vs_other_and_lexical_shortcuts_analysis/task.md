# タスクリスト: V1 (Self vs. Other) — 内部表現・因果経路の共有検証と意味的妥当性の厳密化

## 目的
EmoBank および AIPsy-Affect 4-Split 行動実験（高い行動的連動 $r \approx 0.80 \sim 0.97$）を受け、**「出力が似ているからといって、内部でも同じ表現・同じ因果経路を使っているのか？」**という核心的問い（$\text{Does Similar Behavior imply Shared Representation and Shared Causal Implementation?}$）を、以下の7ステップ（Phase A $\rightarrow$ B $\rightarrow$ C / E1〜E6）の階段構造によって検証する。

---

## タスク一覧

- [x] 1. 実装計画書・理論的定式化の策定 <!-- id: 0 -->
  - [x] 1.1 査読耐性を最大化する7ステップ（Behavior $\rightarrow$ Decodability $\rightarrow$ Geometry $\rightarrow$ Semantic Validity $\rightarrow$ Causality）の階段構造策定 <!-- id: 1 -->
  - [x] 1.2 V1 (Reader vs Self), V2 (Base vs Instruct), V3 (Decodability vs Causality) の役割分担の固定 <!-- id: 2 -->
  - [x] 1.3 ユーザーフィードバックに基づく細部修正（E1〜E6の厳密化、中立な仮説設定）の反映 <!-- id: 3 -->
  - [x] 1.4 全8モデル・全層・フルデータ（EmoBank 1,006件 + AIPsy 2,196件/480ペア）実行可能性と所要時間見積もりの策定 <!-- id: 16 -->

- [ ] 2. Phase A: 全モデル・全層 表現の存在と幾何構造（E1 & E2）の実行 <!-- id: 4 -->
  - [x] 2.1 スクリプト配備: `v1/scripts/run_v1_phase_a_probe.py`（`--is-instruct` 引数未定義エラーを修正） <!-- id: 5 -->
    - E1: EmoBank人間VADへのRidge回帰 ($R^2$)、AIPsyカテゴリへのLogistic回帰 (ROC-AUC, Balanced Acc)、強度への順序回帰
    - E2: Reader $\leftrightarrow$ Self 間の Direct Cross-Decoding, RSA, Orthogonal Procrustes
  - [ ] 2.2 全8モデル（Qwen, LLaMA, Gemma, Mistral × Base/Instruct）× 全層のフルバッチ実行（現在Base 4種＋Qwen Instruct完了、残りInstruct 3種） <!-- id: 6 -->

- [x] 3. Phase B: 全モデル 意味的妥当性の検証・語彙ショートカットの排除（E5） <!-- id: 7 -->
  - [x] 3.1 スクリプト配備: `v1/scripts/run_v1_phase_b_semantic_audit.py` <!-- id: 8 -->
    - Lexical Confound Audit（Jaccard, Edit distance, Emotion lexicon, PPL）の算出
    - 4段階統制対（Minimal pair, Outcome reversal, Paraphrase, Word shuffle）の評価
  - [x] 3.2 全8モデル × 192マッチドペアでの一括実行と結果集計の完了 <!-- id: 9 -->
  - [x] 3.3 レポート生成スクリプトの誤記修正・客観判定化（考察の完全排除、Gemma Base Shuffle +0.021 の正確な反映、Outcome Reversalのファミリー差に応じたラベル化） <!-- id: 17 -->

- [ ] 4. Phase C: 因果回路の検証（E3, E4, E6）の実行 <!-- id: 10 -->
  - [x] 4.1 計算規模・GPU利用の事前見積もりと二段階戦略の策定 <!-- id: 11 -->
  - [x] 4.2 スクリプト配備: `v1/scripts/run_v1_phase_c_causal_patching.py` <!-- id: 12 -->
    - E3: 強度 ＋ 2次元VA方向ベクトルコサイン類似度 $\cos(C^{\text{dir}}_R, C^{\text{dir}}_S)$ による因果マップ
    - E4: 同一ペア単位の差分パッチング $\Delta h_{R, i}$ と用量反応スイープ（$\alpha \in \{-1, 0, 0.5, 1, 1.5\}$）、4大コントロール
  - [x] 4.3 スクリプト配備: `v1/scripts/run_v1_phase_c_targeted_ablation.py` <!-- id: 13 -->
    - E6: 線形混合効果モデル（LMM: $\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$）による因果的特異化／部分解離の交互作用検定
  - [ ] 4.4 因果介入のバッチ実行（代表層または全層夜間バッチ） <!-- id: 14 -->

- [ ] 5. 実験結果の集計・論文執筆への統合（`v3/docs/paper_v1.md`） <!-- id: 15 -->


