# タスク: 論文（iclr2027_conference2.tex）の構成・コード整合性・重複確認および再構成

## 目的
`iclr2027/iclr2027_conference2.tex` について、以下の点を確認・検証・修正案の作成を行う：
1. **コードとの整合性確認**: 実験コード（behavioral, v1, v2, v3, src, configs）および実際の出力結果・設計と記述が一致しているか。
2. **指定構成順序への適合性確認**:
   - 背景 (Introduction / Background)
   - 関連研究 (Related Work)
   - 実験全体の概要 (Overall Experimental Setup / Overview)
   - Behavioral概要 (Behavioral Overview)
   - Behavioral結果 (Behavioral Results)
   - Behavioral考察 (Behavioral Discussion)
   - Behavioralからv1への接続 (Transition: Behavioral to v1)
   - v1概要 (v1 Overview)
   - v1結果 (v1 Results)
   - v1考察 (v1 Discussion)
   - v1からv2への接続 (Transition: v1 to v2)
   - v2概要 (v2 Overview)
   - v2結果 (v2 Results)
   - v2考察 (v2 Discussion)
   - v2からv3への接続 (Transition: v2 to v3)
   - v3概要 (v3 Overview)
   - v3結果 (v3 Results)
   - v3考察 (v3 Discussion)
   - 全体の考察 (General Discussion / Synthesis)
3. **重複の確認と排除**: セクション間での記述の重複、冗長な定義、重複した表・結果記述の洗い出し。

## タスクリスト
- [x] 現状の `iclr2027/iclr2027_conference2.tex` の全体構造・セクション構成の精査
- [x] リポジトリの実装コード（`behavioral`, `v1`, `v2`, `v3`）および設定・最新仕様（`configs/`, `README.md`, `docs/`, `results/`）の確認
- [x] 論文記述とコード・実験結果の整合性チェック（用語、測定指標、モデル、プロトコル、結果数値など）
- [x] 指定構成順序（背景 → 関連研究 → 実験全体の概要 → Behavioral概要/結果/考察/接続 → v1概要/結果/考察/接続 → v2概要/結果/考察/接続 → v3概要/結果/考察/接続 → 全体の考察）との対比・重複箇所の特定
- [x] 実装計画（`implementation_plan.md`）の作成
- [x] 修正方針の提示および「重複のみを修正して」の指示受領
- [x] 重複Aの排除（§3と各ステージ概要の共通基盤重複を簡潔な参照へ集約）
- [x] 重複Bの排除（概要末尾・考察・接続・次概要冒頭の三重論理重複を解消）
- [x] 指定19セクション構成の完全な確立
- [x] 論文結果章用 LaTeX テーブル・結果まとめ生成スクリプト群の実装
  - [x] 1. Behavioral (EmoBank): `scripts/summarize_behavioral_emobank.py`
    - ファミリー別・モデル別 Writer/Reader/Self の VAD 対応表、Base vs Instruct 対比表、内部認知結合度表
  - [x] 2. Behavioral (AIPsy-Affect): `scripts/summarize_behavioral_aipsy.py`
    - RQ1 (Sensitivity), RQ2 (Dose-Response), RQ3 (Specificity), RQ4 (Reader-Self Coupling)
  - [x] 3. V1 (Representation & Causal Sharing): `scripts/summarize_v1_internal_sharing.py`
    - ファミリーごとの Reader vs Self 対応（E1 Decodability, E3 Causal Map, 回路共有度）
  - [x] 4. V2 (Post-training Reorganization): `scripts/summarize_v2_reorganization.py`
    - ファミリーごとの Base vs Instruct 対応（H1a 幾何歪み, H1b ピークシフト, H2 共有度変化, H3 因果再配置・LMM, H4 回復率）
  - [x] 5. V3 (Causal Utilization): `scripts/summarize_v3_causal_utilization.py`
    - Gate 判定表（NO_GO）、時空間 4-Maps 表、独立ファミリー検証再現性マトリックス
  - [x] 6. 統合マスターオーケストレーター: `scripts/generate_paper_results_tables.py`
- [x] スクリプトの実行検証と `iclr2027/tables/` への出力確認（16個のLaTeX表およびMarkdownサマリーの配置完了）
- [x] 論文本文（`iclr2027_conference2.tex`）の各結果セクションへのテーブルインポートと結果解説の記述完了
- [x] 完了確認・Walkthrough（`walkthrough.md`）の作成
