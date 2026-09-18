# V2 コードベース監査タスク (未実装箇所の徹底検証)

- [x] 1. 提示された懸念点（12.1〜12.8, 13）に関する実コードの個別検証 <!-- id: 0 -->
  - [x] 1.1 `affective_empathy_eval.models` (adapters, hooks, registry) の実在確認（解決済みであることを確認）
  - [x] 1.2 `run_v2_recovery_patching.py` の実パッチング実装状況の確認（未実装であることを確認）
  - [x] 1.3 `run_v2_2x2_causal_map.py` および `run_v2_recovery_patching.py` の尤度計算（logits[:81]の切り出しバグを確認）
  - [x] 1.4 `run_v2_2x2_cross_decoding.py` のデータ分割手法（行単位のランダムシャッフルであることを確認）
  - [x] 1.5 `matched_plain` フォーマットの適用状況の確認（未実装であることを確認）
  - [x] 1.6 RQ1/RQ2の評価対象（ValenceのみでありArousal未実装であることを確認）
  - [x] 1.7 差の差（Difference-in-Differences）の計算式の確認（独自定義であることを確認）
  - [x] 1.8 Bootstrap / 統計検定の接続状況の確認（未接続であることを確認）
  - [x] 1.9 旧スクリプト（`run_strict_cross_decoding.py`, `run_introspective_accessibility_test.py` 等）のmock/疑似実装状況の確認（事実であることを確認）
- [x] 2. 監査結果の整理とレポート作成 (`docs/audit_v2_code_status/audit_report.md`) <!-- id: 1 -->
- [x] 3. ユーザーへの詳細報告と改善方針の提示 <!-- id: 2 -->
