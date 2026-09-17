# タスクリスト: クロスファミリー感情認識評価 & 論文ナラティブ改訂

## 1. 感情認識・自己報告のクロスファミリー実測評価
- [x] 統一コホート設定ファイル（`v1/configs/eval_cohort.yaml`）の作成
- [x] 汎用評価スクリプト（`v1/scripts/run_cross_family_recognition.py`）の実装（EmoBank / AIPsy 両対応、パースフォールバック強化）
- [x] サマリー集計スクリプト（`v1/scripts/summarize_cross_family_recognition.py`）の実装
- [x] Qwen2.5-1.5B (Base / Instruct) on EmoBank & AIPsy 評価実行
- [x] Llama-3.2-1B (Base / Instruct) on EmoBank & AIPsy 評価実行
- [x] Mistral-7B (Base / Instruct) on EmoBank & AIPsy 評価実行
- [x] Gemma-2-2B (Base / Instruct) on EmoBank & AIPsy 評価実行
- [x] サマリー表の自動分離生成（EmoBank表とAIPsy表の独立集計）

## 2. 論文（paper2.md / paper3.md）のナラティブ改訂
- [x] 導入部・問題設定の修正（一律SuppressionからRepresentation-to-report Remappingへ）
- [x] 概念図およびリサーチクエスチョン（RQ）の更新
- [x] 付録 C.1 / ベースライン節に8モデル実測テーブルを反映
- [x] 機構解析（Distributed Remapping, Decodability != Causal Leverage）の整合化
- [x] 改訂内容の検証（数値・学術的整合性の確認）
