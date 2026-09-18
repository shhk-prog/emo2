# 実装計画: 再実行前クリティカル修正および逐次保存・レジリエンス強化

## 概要
全再実行前に発見された3大実質的問題（V3 Registry Model ID KeyError、欠落した V1 Phase B 入力、Transformers 下限）および再実行前5大必須修正、逐次保存（Checkpointing / レジリエンス）を迅速かつ堅牢に解決・実装します。

---

## 1. 提案される変更点

### 1.1 V3 ModelRegistry 逆引き修正
- `src/affective_empathy_eval/models/registry.py`:
  - `get_family_by_model_id(self, model_id: str) -> ModelFamilyConfig` メソッドを追加。
  - `get_family(self, key: str)` に、引数が model_id だった場合の逆引き自動解決フォールバックを追加。
- V3 スクリプト群 (`v3/primary/run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py`):
  - `fam_cfg = registry.get_family_by_model_id(model_id)` または `family_id` を渡すように安全改修。

### 1.2 V1 Phase B 入力生成スクリプト作成 (`prepare_v1_phase_b_controls.py`)
- `v1/primary/prepare_v1_phase_b_controls.py` を新規作成。
  - `v1/data/processed/aipsy_4split_all.csv` から、同一 `pair_id` を持つ clinical と neutral をマッチング。
  - 決定論的変換（Paraphrase, Word Shuffle, Outcome Reversal）を実行。
  - `v1/data/processed/v1_e5_semantic_controls.csv` として出力。
- `scripts/run_production_v1.sh` および `src/affective_empathy_eval/run.py` に、本CSVが存在しない場合に自動生成するステップを組み込み。

### 1.3 Transformers 依存下限の引き上げ
- `pyproject.toml` および `requirements.txt`:
  - `transformers>=4.40.0` を `transformers>=4.50.3` に引き上げ。

### 1.4 V1 Phase B の pair-aware CV 化 & 理論的注記
- `v1/primary/run_phase_b.py`:
  - `StratifiedKFold` を `StratifiedGroupKFold(groups=pair_id)` に変更（データリーケージの厳格防止）。
  - Original probe を全データ fit して Paraphrase/Reversal へ適用する部分は「未知pairへの一般化ではなく、Originalで定義したdecision boundaryへのtransformation sensitivityを見る解析」である旨を明記。

### 1.5 V3 実モデル経路を通す Registry Integration Test の追加
- `tests/test_model_registry_and_adapters.py`:
  - 4 family (8モデル) + Mistral 7B の全 model_id から `get_family_by_model_id` が正しく引けるテストを追加。
  - Mock model を用いて V3 の活性化フック・エントリポイントが正常に動作する integration test を追加。

### 1.6 精度・整合性向上
- `v3/primary/run_confirmatory_replication.py`: held-out split に `GroupKFold(groups=pair_id)` を使用。
- `v3/primary/run_rq2_spatiotemporal_maps.py`: 出力名を `v3_discovery_spatiotemporal_maps_*.json` に変更し、全サンプル数 $N$ と介入サンプル数 $N_{\mathrm{intervene}}$ を manifest に記録。
- V3 Topic Control の位置づけ（非特異的除外統制）の注記強化。
- `v1/primary/run_phase_c.py` 等に残る `sys.path.insert()` の整理。

### 1.7 逐次保存・レジリエンス (Checkpointing)
- `run_behavioral_emobank.py` / `run_behavioral_aipsy.py`: 10サンプルごと、および各サンプル処理時に中間チェックポイントを保存。
- V1 / V2 / V3: レイヤー単位・モデル単位で処理完了ごとに JSON / CSV を逐次書き出し、中断時にも途中経過を保持。

---

## 2. 検証計画
- `pytest -q`: 全テストの合格確認。
- `prepare_v1_phase_b_controls.py` の実行確認と出力 CSV 検証。
- `bash scripts/run_production_all.sh cpu --dry-run`: 全パイプラインの結合テスト完走確認。
