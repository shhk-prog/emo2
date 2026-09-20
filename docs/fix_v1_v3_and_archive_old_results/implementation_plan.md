# V1 / V3 パイプライン修正および旧結果退避計画

## 概要
研究の測定妥当性と再現性を高めるため、V1 (Phase A, Phase B, E4, E6) および V3 (RQ1, Gate, Confirmatory Site Generation) のコードを修正し、再実行が必要となる旧結果を削除せず `old_results/` ディレクトリへ安全に退避・アーカイブします。
また、Legacy コードの二重 Softmax 呼び出しを排除し、Likelihood 記述をコードの厳密な定義（joint conditional sequence log-likelihood）と一致させます。

---

## 修正対象と変更内容

### 1. 旧結果の退避（Archive to `old_results/`）
- `old_results/` ディレクトリをプロジェクトルートに作成。
- **V1 Phase A**: EmoBank などの結果は温存し、各モデル配下の AIPsy 分類結果 `e1_aipsy_classification.csv` のみを `old_results/v1_phase_a/{model_prefix}/` に移動。
- **V1 Phase B**: 従来の Reader-only 結果（`v1/results/derived/v1_phase_b/*`）を `old_results/v1_phase_b/` に移動。
- **V1 E6**: 既存の E6 結果ファイルがあれば `old_results/v1_e6/` に移動。
- **V3 RQ1 / Gate / RQ2 / RQ3 / Confirmatory**:
  - `v3/results/raw/v3_rq1_results.json`
  - `v3/results/raw/manifest_rq1_qwen.json`
  - `v3/results/derived/v3_gate_decision.json`
  - その他もしあれば `frozen_confirmatory_sites.json`, `v3_rq2_results.json`, `v3_rq3_results.json`, `v3_confirmatory_*.json` を `old_results/v3/` に移動。

### 2. V1 Phase A: AIPsy Primary 解析および Cross-decoding Scaler の修正
- [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py):
  - **AIPsy Primary 解析**:
    - 従来の 8 emotion category 分類を Primary から外し、`clinical vs matched neutral`（バイナリ分類、`split == "clinical"` vs `split == "neutral"`）を Primary に設定。
    - グループは `pair_id`。`GroupKFold`（または `StratifiedGroupKFold`）によりペア単位の held-out 評価（balanced accuracy, f1_macro, roc_auc）を行う。
    - 強度解析: `triplet_id` が存在する行のみを抽出し、`none < moderate < peak` (0, 1, 2) の順序関係解析（Spearman $\rho$ / 順序相関）を別ファイル `e1_aipsy_intensity.csv` として出力。
    - 従来の 8 emotion category 分類は Secondary 解析として `e1_aipsy_emotion_secondary.csv` に出力。
  - **Cross-decoding Scaler**:
    - `evaluate_cross_decoding_and_geometry` において、Reader $\to$ Self の転移予測時は Reader 訓練データで fit した `scaler_r` を Self のテストデータにも適用 (`clf_r.predict(scaler_r.transform(HS_te))`)。
    - Self $\to$ Reader の転移予測時も同様に Self 訓練データで fit した `scaler_s` を Reader のテストデータに適用 (`clf_s.predict(scaler_s.transform(HR_te))`)。
    - 従来の「task ごとに個別 fit した scaler」による予測も Secondary 指標（`pred_r_to_s_task_scaled` 等）として併せて記録。

### 3. V1 Phase B: Reader / Self 両条件の実行と出力先分離
- [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py):
  - 出力ディレクトリを `--task-type` に基づき `v1/results/derived/v1_phase_b/{task_type}/{model_prefix}/` に自動分離し、Reader と Self で CSV/manifest が上書きされないように修正。
- [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py):
  - Phase B の実行部で、`--task-type reader` と `--task-type self` の双方を順次実行するように更新。

### 4. V1 E6: Valence だけでなく VA を Primary に設定
- [`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py):
  - 各介入ペアについて以下を算出・保存：
    - `impact_v = abs(delta_v)`
    - `impact_a = abs(delta_a)`
    - `impact_va = sqrt(delta_v^2 + delta_a^2)`
  - Primary metric を `impact_va` に統一し、LMM 交互作用検定および t-test fallback を `impact_va` を主目的変数として実行。
  - Secondary として `impact_v`, `impact_a` についても同様の検定を実行し結果辞書に保持。

### 5. V1 E4: 多重比較の整理
- [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py):
  - E3 で同定された peak layer かつ $\alpha=1.0$ の条件を事前固定の Primary Confirmatory 条件とし、フラグ `is_confirmatory` を付与。
  - その他すべての探索的条件（各 layer, $\alpha$, V/A 軸）に対して Benjamini-Hochberg による FDR 補正後 p 値（`p_fdr_V`, `p_fdr_A`, `aligned_p_fdr_V`, `aligned_p_fdr_A`）を算出し結果 DataFrame に付与。

### 6. V3 RQ1 & Gate: train/test 比率の config 一致と slope 閾値の config 化
- [`configs/v3_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml):
  - `gate_criteria` に `min_sufficiency_slope: 0.1` を明示。
- [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py):
  - ハードコードされた `half_pairs = len(unique_pairs) // 2` を削除。
  - `config["dataset"].get("train_ratio", 0.7)` に基づき 70:30 の Group split を実行。
  - Gate 判定関数 `evaluate_go_no_go_gate` でハードコードされた `0.1` を `gate_cfg.get("min_sufficiency_slope", 0.1)` に置き換え。

### 7. V3 Confirmatory site 生成・参照の厳密化
- [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py):
  - RQ2 単独終了時に `frozen_confirmatory_sites.json` を確定出力するのを廃止。
- [`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py):
  - RQ3 完了時に、RQ1 の Gate 通過 site (`sufficiency_relative_depth`)、RQ2 の causal peak site (`temporal_relative_depth`)、RQ3 で実際に同定した mediator site (`mediator_relative_depth`) を統合した `frozen_confirmatory_sites.json` を確定出力。
- [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py):
  - 本番実行時に `frozen_confirmatory_sites.json` が存在しない場合は、推測や config fallback を行わず即座に `FileNotFoundError` で停止。

### 8. Likelihood 記述と Legacy コードの整理
- [`affective_empathy_eval/likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py) および関連ドキュメントにおいて、候補対数尤度の定義が `joint conditional sequence log-likelihood`（完全な条件付き対数尤度和）であることを明記。
- [`v3/scripts/legacy/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/legacy/README.md) に `DO NOT USE FOR PAPER RESULTS` と明記。
- Legacy スクリプト内の `compute_expected_va(probs, ...)` 二重 Softmax 呼び出しを削除・置換。

---

## 検証計画

### 1. 単体・構文テスト
- 既存テストおよび修正対象スクリプトの構文チェック・ユニットテストを実行:
  ```bash
  .venv/bin/python -m pytest tests/test_production_entrypoints.py tests/test_v1_token_and_probe_alignment.py -v
  ```

### 2. パイプラインドライラン検証
- V1 Phase A, B, C, E6 のドライラン実行:
  ```bash
  .venv/bin/python v1/primary/run_phase_a.py --dry-run
  .venv/bin/python v1/primary/run_phase_b.py --task-type reader --dry-run
  .venv/bin/python v1/primary/run_phase_b.py --task-type self --dry-run
  .venv/bin/python v1/primary/run_phase_c.py --dry-run
  .venv/bin/python v1/primary/phase_c/run_e6_specialization.py --dry-run
  ```
- V3 RQ1, RQ2, RQ3, Confirmatory のドライラン実行:
  ```bash
  .venv/bin/python v3/primary/run_rq1_state_induction.py --dry-run
  .venv/bin/python v3/primary/run_rq2_spatiotemporal_maps.py --dry-run
  .venv/bin/python v3/primary/run_rq3_path_mediation.py --dry-run
  .venv/bin/python v3/primary/run_confirmatory_replication.py --dry-run
  ```

### 3. 退避ディレクトリ確認
- `old_results/` 配下に旧結果が整然と格納されていることを確認。
