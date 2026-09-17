# タスク管理: paper3.md 総合改訂（不整合解消と厳密化）

## 目的
ユーザーからの詳細な査読・整合性レビューに基づき、`v3/docs/paper3.md` の11項目の重要不整合を解消し、実験データ（CSV正本）と完全に一致した学術的に極めて厳密な論文原稿へ改訂する。

## タスク一覧
- [x] 計画策定と詳細確認 (`implementation_plan.md`)
- [x] 1952行以降の「第II部 統合再編」の確認および削除・必要図表の本文移行（Table 1, 2, 3 の本文統合）
- [x] Related Work の文献年および全称命題の修正（Hase et al. 2023, Maheswaran & Desarkar 2026）
- [x] 「Wasserstein-2」表記の排除と 2D Joint OT (Manhattan ground cost) への統一
- [x] Probe necessity のランダム方向数（Full-layer N=20, Focused N=100）および表現（No evidence for local necessity）の統一
- [x] Generation-time 全層表（Table A2a: Exploratory 15-pair vs Table A2b: Focused 39-pair）の分離
- [x] Multi-layer Residual 条件（L18, L20, L24, L20+24, L18+20+24, L18-24）と数値の統一（架空条件 L22+24 等の完全排除）
- [x] Llama-3.2-1B-Instruct 追試記述の厳密化（最大+0.73% at L4 Attn、過度の一般化排除）
- [x] 「確証的」「バイアスのない」の用語修正（Exploratory full-layer screen / Focused full-cohort effect-size evaluation）
- [x] 「因果レバー皆無」等の過剰表現の是正（tested local matched-substitution では局所因果回復を示さなかった）
- [x] Appendix 体系の再編・整合（重複 Appendix C の削除、Appendix A〜G の一意連番化）
- [x] 変更内容の確認とドキュメント作成 (`walkthrough.md`)
