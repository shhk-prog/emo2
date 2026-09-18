# タスクリスト: 本番用データ全件一括実行パイプラインの整備と実行ガイド

## 目的
Behavioral, V1, V2, V3 の各ステージにおいて、Primary 4小型family (Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B) を「本論文に使える本番データ規模」で一気に実行できるコード（スクリプト・ランナー）を整備し、実行コマンドを提供する。

---

## 本番データ仕様
1. **Behavioral Stage**:
   - EmoBank 3-Way VAD: 1,000 刺激 (全件)
   - AIPsy-Affect 4-Split: 480 刺激 (全件)
   - 対象: 4 family $\times$ 2 条件 (Base, Instruct)
2. **V1 Stage**:
   - Phase A: EmoBank 1,000 刺激 + AIPsy 480 刺激 (全層デコード・幾何解析)
   - Phase B: 400 刺激 (全件マッチド統制ペア)
   - Phase C (E3 Causal Map / E4 Interchangeability): AIPsy 480 刺激 (全件)
   - Phase C E6 (Double Dissociation): AIPsy 480 刺激 (全件)
   - 対象: 4 family $\times$ 2 条件 (Base, Instruct)
3. **V2 Stage**:
   - RQ1/RQ2: 1,000 刺激 (700 train / 300 held-out test, 28-32全層)
   - RQ3: 1,000 刺激 (全層因果マッピング・LMM)
   - RQ4: 1,000 刺激 (分布回復パッチング・Wasserstein/EMD)
   - 対象: 4 family (Base & Instruct 比較)
4. **V3 Stage**:
   - RQ1: 1,000 刺激 (Gate 判定: Specificity, Necessity, Slope, Topic Control)
   - RQ2: 1,000 刺激 (4-Maps 時空間幾何・因果全層探索)
   - RQ3: 1,000 刺激 (Discovery 500 / Confirmation 500 全件 Path Mediation)
   - Step 7 Confirmatory: 4 family 全モデル (全件追試)

---

## タスク一覧
- [x] 1. **ドキュメント準備**:
  - [x] `docs/full_production_execution_pipeline/task.md` 作成
  - [x] `docs/full_production_execution_pipeline/implementation_plan.md` 作成
- [x] 2. **本番データ規模（全件・サブサンプル撤廃）のコード改修**:
  - [x] `v1/primary/run_phase_c.py`: `--limit` default=0 (全件)
  - [x] `v1/primary/phase_c/run_e6_specialization.py`: `--limit` default=0 (全件)
  - [x] `v3/primary/run_rq2_spatiotemporal_maps.py`: `--subsample` default=0 (全件)
  - [x] `v3/primary/run_rq3_path_mediation.py`: `--subsample 0` で全件使用に対応
  - [x] `v3/primary/run_confirmatory_replication.py`: `--subsample 0` で全件使用に対応
  - [x] `src/affective_empathy_eval/run.py`: 本番実行時に各スクリプトへ制限なし（フルデータ）を確実に渡すように保証
- [x] 3. **4小型family一括実行用本番スクリプトの作成**:
  - [x] `scripts/run_production_all.sh` の作成
  - [x] `scripts/run_production_behavioral.sh` の作成
  - [x] `scripts/run_production_v1.sh` の作成
  - [x] `scripts/run_production_v2.sh` の作成
  - [x] `scripts/run_production_v3.sh` の作成
  - [x] ログ出力、GPU指定、タイムスタンプ記録、エラーハンドリングを完備
- [x] 4. **検証とコマンド案内**:
  - [x] `--dry-run` による全ステージ・全4family実行検証
  - [x] GPU環境での本番実行コマンドおよび所要時間見積もりの提示
  - [x] `walkthrough.md` の作成と保存
