# タスク定義: V1 E6タスク選択性サイト選定と全再実行直前の最終整合化

## 目的
本番全再実行（Primary 4-Family: Qwen, Llama, Gemma, OLMo）を開始する直前の最終課題として、V1 E6 の科学的妥当性向上（タスク選択性差 $S_R(l), S_S(l)$ による site 選定、人工的ピーク置換の廃止、Negative Result の安全処理、フォールバックの完全撤去）、用語の完全統一（Task-Specific Causal Specialization）、V3 RQ2 への Discovery 明記、および V2 旧スクリプトの legacy 隔離を実施する。

## タスク項目
- [x] V1 E6: タスク選択性差コントラストによる site 選定の実装
  - $S_R(l) = C_R(l) - C_S(l)$
  - $S_S(l) = C_S(l) - C_R(l)$
  - 同一層または最大選択性 $\le 0$ の場合の人為的第2ピーク差し替えを廃止し、No-Go / negative result として判定・記録して安全に終了
  - E3 Discovery CSV が存在しない場合の heuristic fallback（$0.5(L-1), 0.6(L-1)$）を完全撤去し、`FileNotFoundError` を送出
- [x] 表記統一: "Double Dissociation" をすべて "Task-Specific Causal Specialization"（または "Causal Specialization / Partial Dissociation"）へ統一
  - `v1/README.md`
  - `src/affective_empathy_eval/run.py`
  - `scripts/run_production_v1.sh`
  - `v1/primary/phase_c/run_e6_specialization.py`
  - `v1/primary/phase_c/summarize_phase_c.py`
- [x] V3 RQ2: Discovery Role の明記
  - 出力 JSON および manifest に `analysis_role = "discovery"` を明記
- [x] V2 Legacy 整理: 旧 Qwen 専用スクリプトおよび config を `v2/scripts/legacy/`, `v2/configs/legacy/` へ隔離
- [x] Results 初期化状態の確認（`.gitkeep` のみ）
- [x] ドキュメント保存 (`docs/e6_selectivity_and_final_refinements/`)
