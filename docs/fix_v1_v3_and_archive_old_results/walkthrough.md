# 修正内容の確認 (Walkthrough): V1 / V3 パイプライン修正および旧結果退避

本ドキュメントは、V1 Phase A/B/C/E6 および V3 (RQ1〜RQ3、Confirmatory) パイプラインの統計的・因果的評価の厳密化、旧結果の安全な退避、および Legacy コードの二重 Softmax 呼び出し是正の実施内容と検証結果をまとめたものです。

---

## 1. 実施した変更内容の概要

### 1.1 再実行対象の旧結果の退避 (`old_results/`)
上書き禁止および実験再現性維持の観点から、修正対象となる旧解析結果を削除せず、`old_results/` ディレクトリへ安全に移動・退避しました。EmoBank等の再利用可能な原データ・基本特徴量・中間結果は温存しています。
- `old_results/v1_phase_a/`: 各モデルディレクトリ配下の旧 `e1_aipsy_classification.csv`
- `old_results/v1_phase_b/`: 旧 Reader 単一条件の Phase B 出力
- `old_results/v3/raw/`: 旧 `v3_rq1_results.json`, `manifest_rq1_qwen.json`
- `old_results/v3/derived/`: 旧 `v3_gate_decision.json`

### 1.2 V1 Phase A: Direct cross-decoding & AIPsy 解析修正 ([`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py))
- **Direct cross-decoding の scaler 修正**:
  - `Reader -> Self`: Reader の訓練データで fit した `StandardScaler` を Reader (train/test) と Self (test) の双方に適用。
  - `Self -> Reader`: Self の訓練データで fit した `StandardScaler` を Self (train/test) と Reader (test) の双方に適用。
  - 各タスク専用 scaler による評価は Secondary 指標として並行記録。
- **AIPsy Primary 解析の厳密化**:
  - 8 emotion category 分類を Primary から外し、`clinical vs matched neutral`（2値分類、GroupKFold with `group="pair_id"`）を Primary 指標として `e1_aipsy_classification.csv` に保存。
- **AIPsy 強度解析の分離**:
  - `triplet_id` が存在する行のみを抽出し、`none < moderate < peak` の順序解析・強度分離解析として `e1_aipsy_intensity.csv` に新規出力。
- **Emotion decoding の Secondary 化**:
  - 従来の 8 emotion category 多クラス分類を Secondary 解析として `e1_aipsy_emotion_secondary.csv` に出力。

### 1.3 V1 Phase B: Self 条件の追加と出力先分離 ([`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py), [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py))
- `--task-type` 引数を受け取り、結果の出力先を `v1/results/derived/v1_phase_b/{task_type}/{model_prefix}/` に完全に分離。
- `run.py` 内のパイプライン実行ロジックにおいて、`task-type=reader` と `task-type=self` の両条件を自動的に順次実行するように更新。

### 1.4 V1 E6: Primary 指標の統合 ([`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py))
- 個別変位 $\Delta V$, $\Delta A$ に加えて、合成変位ベクトルノルム $\text{impact}_{VA} = \sqrt{\Delta V^2 + \Delta A^2}$ を算出。
- Primary metric を `impact_va` に統一し、LMM 交互作用検定（Domain × Intervention）およびペア比較を実施。
- `impact_v`, `impact_a` も Secondary 指標としてモデル出力とセル平均を保存。

### 1.5 V1 E4: 多重比較補正の厳密化 ([`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py))
- Primary confirmatory condition（`E3 peak layer × alpha=1.0`）に `is_confirmatory = True` フラグを付与。
- 全介入条件（layer × alpha × dimension）に対して Benjamini-Hochberg 法による FDR 補正後 p 値（`p_fdr_V`, `p_fdr_A`, `aligned_p_fdr_V`, `aligned_p_fdr_A`）を算出し、結果表に出力。

### 1.6 V3 RQ1 & Gate 判定修正 ([`configs/v3_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml), [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py))
- `configs/v3_experiments.yaml` の `gate_criteria` に `min_sufficiency_slope: 0.1` を明示定義。
- `run_rq1_state_induction.py` 内のハードコードされた `half_pairs`（50:50分割）を撤廃し、config の `train_ratio: 0.7`（70:30）による GroupShuffleSplit へ修正。
- Gate 判定の slope 閾値（0.1）を config から動的に読み込むように統一。

### 1.7 V3 Confirmatory site 生成・読込修正
- [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py):
  - RQ2 単独での `frozen_confirmatory_sites.json` 確定出力を廃止し、中間アーティファクト `v3_rq2_causal_sites.json` として出力。
- [`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py):
  - RQ3 完了時に、RQ1 の Sufficiency site、RQ2 の Causal site、RQ3 の Mediator site を一貫して統合した最終確定版 `frozen_confirmatory_sites.json` を生成。
- [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py):
  - 本番実行時に `frozen_confirmatory_sites.json` が存在しない場合は、推測やデフォルト値へのフォールバックを行わず、即座に `FileNotFoundError` を送出するガードを実装。

### 1.8 Legacy コードの二重 Softmax 是正および Likelihood 表記統一
- `compute_expected_va` は未正規化の log-scores を受け取って log-sum-exp 安定化 Softmax を適用する仕様であるにもかかわらず、legacy スクリプトで Softmax 済み `probs` が渡されていた問題を完全是正（未正規化 `log_liks` を渡すように修正）。
  - 修正対象: [`v3/scripts/legacy/run_v3_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/legacy/run_v3_state_induction.py), [`v3/scripts/legacy/run_v3_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/legacy/run_v3_spatiotemporal_maps.py), [`v3/scripts/legacy/run_v3_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/legacy/run_v3_confirmatory_replication.py)
- [`v3/scripts/legacy/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/legacy/README.md) に `DO NOT USE FOR PAPER RESULTS` の警告を明記。
- Likelihood 表記を `joint conditional sequence log-likelihood` に統一。

---

## 2. 検証結果

### 2.1 自動テスト (pytest)
既存のテストスイートを実行し、全テストがパスすることを確認しました：
```bash
.venv/bin/pytest tests/test_production_entrypoints.py tests/test_v1_token_and_probe_alignment.py tests/test_likelihood.py tests/test_v3_interventions_sanity.py tests/test_confirmatory_pipeline.py
# 結果: 32 passed, 1 deselected in 6.51s

.venv/bin/pytest tests/test_v3_causal_extensions.py tests/test_refinement_suite.py tests/test_v1_refinements.py
# 結果: 16 passed in 4.48s
```

### 2.2 ドライラン検証
修正した各スクリプトのドライラン（`--dry-run`）を仮想環境 `.venv` 内で実行し、全て exit code 0 で正常終了することを確認しました：
1. **V1 Phase A**: `python v1/primary/run_phase_a.py --dry-run` $\to$ PASS
2. **V1 Phase B (Reader & Self)**:
   - `python v1/primary/run_phase_b.py --dry-run --task-type reader` $\to$ PASS
   - `python v1/primary/run_phase_b.py --dry-run --task-type self` $\to$ PASS
3. **V1 Phase C / E6**:
   - `python v1/primary/phase_c/run_e6_specialization.py --dry-run` $\to$ PASS
   - `python v1/primary/run_phase_c.py --dry-run` $\to$ PASS
4. **V3 RQ1 / RQ2 / RQ3 / Confirmatory**:
   - `python tests/run_all_v3_dryruns.py` により、RQ1 $\to$ RQ2 $\to$ RQ3 $\to$ Confirmatory のパイプライン連係がドライランで完全動作することを確認。
