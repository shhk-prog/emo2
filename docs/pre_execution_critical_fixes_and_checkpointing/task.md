# タスクリスト: 再実行前クリティカル修正および逐次保存・レジリエンス強化

## 目的
本番全再実行を確実に成功させるため、発見された3つの実質的問題（V3 Registry Model ID、欠落したV1 Phase B入力生成、Transformers依存下限）、および再実行前5大必須修正、逐次保存（Checkpointing / レジリエンス）を実装・検証する。

---

## タスク一覧

### 1. 必須修正事項 (Critical Fixes)
- [x] **1.1 V3 ModelRegistry逆引き修正**:
  - `ModelRegistry` に `get_family_by_model_id(model_id)` を追加。
  - `get_family(key)` にも model_id による自動フォールバックを実装。
  - `get_model_adapter` でも `ModelFamilyConfig` インスタンスを直接受容できるように強化。
  - `v3/primary/run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py` で `model_id` / `family_id` を安全に解決するよう修正。
- [x] **1.2 V1 Phase B 入力生成スクリプト作成**:
  - AIPsy (`v1/data/processed/aipsy_4split_all.csv`) から決定論的変換（Minimal Pair, Paraphrase, Word Shuffle, Outcome Reversal）を施す `v1/primary/prepare_v1_phase_b_controls.py` を新規作成。
  - `v1/data/processed/v1_e5_semantic_controls.csv` (192ペア) を生成・配置。
  - `scripts/run_production_v1.sh` および `affective_empathy_eval/run.py` で Phase B 実行前に本前処理スクリプトを自動チェック・実行するよう統合。
- [x] **1.3 Transformers 依存下限の引き上げ**:
  - `pyproject.toml` および `requirements.txt` の `transformers>=4.40.0` を `transformers>=4.50.3` に引き上げ（Gemma 3 / OLMo 2 サポート）。
- [x] **1.4 V1 Phase B の pair-aware CV 化**:
  - `v1/primary/run_phase_b.py` の `evaluate_probe_accuracy()` を `StratifiedGroupKFold(groups=pair_id)` に変更。
  - Original probe を全データ fit して Paraphrase/Reversal へ適用する部分は「未知pairへの一般化ではなく、Originalで定義したdecision boundaryへのtransformation sensitivityを見る解析」である旨を明記。
- [x] **1.5 V3 実モデル経路を通す Registry Integration Test の追加**:
  - 4 family (Base/Instruct 計8モデル + Mistral 7B) の全 model_id が正しく family config を解決できるテストを追加。
  - Mock model を用いて V3 の本番エントリポイントが正常に動作する integration test を追加。

### 2. 精度・整合性向上 (Enhancements)
- [x] **2.1 V3 Confirmatory の held-out split を pair-aware (GroupKFold) に統一**:
  - `v3/primary/run_confirmatory_replication.py` で `pair_id` が存在する場合に `GroupKFold` を使用。
- [x] **2.2 V3 RQ2 4-Map 出力ファイル名を Discovery と明記**:
  - `v3_spatiotemporal_maps_*.json` $\rightarrow$ `v3_discovery_spatiotemporal_maps_*.json` に変更。
- [x] **2.3 V3 RQ2 因果評価サンプル数の manifest 保存**:
  - 全サンプル数 $N_{\mathrm{total}}$ と、実際に介入・評価に用いたサンプル数 $N_{\mathrm{intervene}}$ を結果辞書に分離記録。
- [x] **2.4 V3 Topic Control の位置づけ整理**:
  - Self（VA shift）と Topic（TVD）の指標特性の違いを考慮し、非特異的摂動の除外統制（Control）としての解釈をログ・docstring に明記。
- [x] **2.5 V1 Phase C の sys.path.insert 整理**:
  - `v1/primary/phase_c/run_e3_causal_map.py` および `run_e4_interchangeability.py` の `sys.path.insert()` を完全除去し、クリーンなサブプロセス実行ラッパーに改修。

### 3. 逐次保存・レジリエンス (Checkpointing / Fault Tolerance)
- [x] **3.1 Behavioral ステージの逐次保存**:
  - `run_behavioral_emobank.py` および `run_behavioral_aipsy.py` で 10サンプルごとの定期保存および既存チェックポイントからの自動レジューム（Resume）を導入。
- [x] **3.2 V1 / V2 / V3 各ステージのレイヤー単位・モデル単位チェックポイント保存**:
  - V1 Phase A: レイヤー完了ごとの即時 CSV フラッシュ。
  - V2 RQ1〜RQ4: ファミリー完了ごとの即時 JSON 保存および完了済みファミリーの自動スキップ（レジューム）。
  - V3 RQ2, RQ3, Confirmatory: 出力ファイル存在時の自動スキップ・レジューム。

### 4. 検証と最終報告
- [x] `pytest` による単体テストおよび追加テストの実行（52 passed）。
- [x] `bash scripts/run_production_all.sh cpu --dry-run` の再実行確認（完走・Exit Code 0）。
- [x] `docs/pre_execution_critical_fixes_and_checkpointing/walkthrough.md` の作成。
