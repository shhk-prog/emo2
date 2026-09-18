# 実装および修正確認レポート (Walkthrough)

## 概要
全再実行直前の最終リファインメントとしてご指摘いただいた、研究上・方法論上の重要修正（V1 E6 の Discovery/Confirmation 分離およびサイト動的選定、V2 delta_delta_cross 計算式の対称化、V2 dry-run の真の軽量化、Phase C キャッシュのハッシュメタデータ検証、Procrustes の PCA 安定化、旧コード・設定の legacy 隔離）をすべて実装し、単体テスト（54 passed）および全4ステージ統合パイプライン dry-run の高速完走（3分44秒）を確認しました。

---

## 主な変更点

### 1. V1 E6: Discovery Site 決定 $\rightarrow$ Confirmation Pairs のみでの Targeted Ablation 検定
- **動的サイト選定 (`select_sites_from_e3`)**:
  - `v1/primary/phase_c/run_e6_specialization.py` において、`e3_causal_map.csv` の Discovery 結果（`discovery_mag_reader`, `discovery_mag_self`）から Reader peak および Self peak を動的に同定。
  - 両ピークが同層になった場合でも Self 側を runner-up に選出する安全機構を実装。
- **完全な Confirmation 分離 (`--split-eval confirmation`)**:
  - `pair_id` に基づく 50/50 Group split を適用し、Discovery pairs（50%）で選定したサイトに対し、**完全に独立した Confirmation pairs（50%）のみ**を用いて 2x2 Targeted Ablation および LMM 交互作用検定を実行するよう改修。
- **用語とログの表現緩和**:
  - 「Double Dissociation」から「Task-specific Causal Specialization / Partial Dissociation」へ統一。
  - 出力は `e6_specialization_trials.csv`（主出力）と後方互換用の `e6_double_dissociation_trials.csv` の双方を安全に保存。

### 2. V2 RQ2: delta_delta_cross 計算式の対称化
- `v2/primary/run_rq1_rq2_cross_decoding.py`:
  - 従来の非対称な式から、双方向 cross-decoding の変化差である $(I_{R \to S} - B_{R \to S}) - (I_{S \to R} - B_{S \to R})$ に修正。
  - Reader $\to$ Self の cross-decodability 変化と Self $\to$ Reader の変化の方向差として幾何学的整合性を確保。

### 3. V2 Procrustes: PCA (k=64) による次元削減と安定化
- `src/affective_empathy_eval/geometry.py` (`eval_held_out_procrustes`):
  - 高次元（$D \approx 1536 \sim 2048$）・低サンプル（$N \approx 700$）における rank-deficiency および null space の回転不定性を回避するため、train split から結合主成分基底（$k=64$）を fit し、共通 PCA 部分空間へ射影してから直交回転行列 $R$ を求める構造へ改修。
  - SVD 計算が劇的に高速化かつ数値的に安定化。

### 4. V2 dry-run の真の軽量化（90秒超 $\rightarrow$ 7.5秒）
- `v2/primary/run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`:
  - `--dry-run` 指定時に、データセットサイズを $N \le 32$、層数を $\min(L, 4)$、次元数を $\min(D, 64)$、Bootstrap 反復数を $10$ に縮小。
  - RQ1/RQ2: 90秒超 $\rightarrow$ 7.5秒
  - RQ3: 7.4秒
  - RQ4: 5.6秒
  - スモークテストとして真に軽量で実用的なものになりました。

### 5. V1 Phase C キャッシュのハッシュメタデータ検証
- `v1/primary/run_phase_c.py`:
  - 単なるサンプル数判定 (`baselines_n192.npz`) を廃止。
  - `model_id`, `git_commit`, `dataset_hash`, `prompt_hash`, `candidate_schema`, `intervention_position`, `tokenizer` を含む `_meta.json` を生成。
  - キャッシュロード時に厳格な一致検証を行い、不一致またはメタデータ欠落時は自動的に再計算＆最新メタデータで保存する仕組みを構築。

### 6. V3 RQ2/RQ3: サンプル数 manifest 保存 & ファイル名明示化
- `v3/primary/run_rq2_spatiotemporal_maps.py`:
  - 出力ファイル名を `v3_discovery_spatiotemporal_maps_{fam_key}.json` と明示。
  - `manifest_rq2_{fam_key}.json` を出力し、`n_dataset_total` と実際に介入へ使った `n_intervention_samples` を明記。
- `v3/primary/run_rq3_path_mediation.py`:
  - `manifest_rq3_{fam_key}.json` を出力し、サンプル規模と attenuation 比率を追跡可能に記録。

### 7. 不要・旧ファイルの legacy / archive 隔離
- `v1/configs/models.yaml` $\rightarrow$ `v1/configs/legacy/models.yaml` へ移動（唯一の正本は `configs/models.yaml`）。
- 未使用だった `v2/src/` $\rightarrow$ `v2/src_legacy/`、`v3/src/` $\rightarrow$ `v3/src_legacy/` へ隔離（唯一のアクティブ実装は `src/affective_empathy_eval`）。
- ルートの `plots/` 内の旧図 $\rightarrow$ `docs/archive/plots/` へ隔離。

---

## 検証結果

1. **単体テスト (`pytest -q`)**:
   ```
   54 passed in 7.68s
   ```
   - 新規追加した Procrustes PCA テストや E6、キャッシュロジックを含む全テストが通過。

2. **全ステージ統合 dry-run (`bash scripts/run_production_all.sh cpu --dry-run`)**:
   ```
   >>> [1/4] Behavioral: 2 seconds
   >>> [2/4] V1 Pipeline: ~2.5 minutes (8 models x Phase A, B, C, E6, Summarize)
   >>> [3/4] V2 Pipeline: ~20 seconds (RQ1/RQ2, RQ3, RQ4 across 4 families)
   >>> [4/4] V3 Pipeline: 23 seconds (RQ1, RQ2, RQ3, Replication)
   Total Wall Time: 3m 44s
   ALL PRODUCTION STAGES COMPLETED SUCCESSFULLY!
   ```

3. **results ディレクトリ初期状態の確認**:
   - `results/raw/`, `results/derived/`, 各ステージの `results/` は `.gitkeep` のみのクリーン状態であることを確認済み。
