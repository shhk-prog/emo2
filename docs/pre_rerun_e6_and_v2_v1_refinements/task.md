# タスクリスト: 全再実行直前 E6 Discovery/Confirmation 分離・V2 整合化・キャッシュ厳格化

## 目的
本番全再実行を完全に盤石な状態にするため、ユーザーから指摘された3大必須修正（V1 E6 の E3 Discovery site $\rightarrow$ Confirmation 検定化、V2 delta_delta_cross 式の修正、V2 dry-run の大幅軽量化）、およびキャッシュ検証・Procrustes PCA・古いファイルの整理を完遂する。

---

## タスク一覧

### 1. 必須修正 (Priority 1)
- [ ] **1.1 V1 E6 の Discovery/Confirmation 分離と Site 選択**:
  - `v1/primary/phase_c/run_e6_specialization.py`:
    - 50/50 Group split (`pair_id` ベース) を行い、Discovery 側で E3 的な Reader peak / Self peak 層を同定（または Phase C の結果から解決）。
    - 決定した site に対し、完全に独立した Confirmation pairs のみで 2x2 targeted ablation を検定。
    - 名称・ログ・docstring を「Double Dissociation」から「Causal Specialization / Partial Dissociation」に表現緩和・統一。
- [ ] **1.2 V2 RQ2 delta_delta_cross 式の修正**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`:
    - $(I_{R \to S} - B_{R \to S}) - (I_{S \to R} - B_{S \to R})$（双方向 cross-decoding の変化差）に修正し、幾何学的対称性と一貫性を確保。
- [ ] **1.3 V2 dry-run の真の軽量化**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`:
    - dry-run 時に $N \le 32$, `hidden_dim <= 64`, `layers <= 4` に縮小し、90秒 $\rightarrow$ 数秒で終わる超軽量スモークテスト化。

### 2. 方法論的・運用上の重要改善 (Priority 2)
- [ ] **2.1 V1 Phase C キャッシュの hash metadata 検証**:
  - `v1/primary/run_phase_c.py`:
    - キャッシュに `meta.json` を保存（`model_id`, `git_commit`, `dataset_hash`, `prompt_hash`, `candidate_schema`, `intervention_position`, `tokenizer`）。
    - 読み込み時にハッシュを検証し、変更があれば自動無効化・再計算。
- [ ] **2.2 V2 Procrustes の PCA (50〜64次元) 安定化**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py` (Procrustes):
    - 高次元・低サンプルによる rank-deficiency と null space 回転不定性を回避するため、train split で fit した PCA（$k=64$ または $\min(N_{\mathrm{train}}-1, 64)$）を通してから直交回転を算出。
- [ ] **2.3 古い config / 未使用 src / 旧 plots の legacy 隔離**:
  - `v1/configs/models.yaml` $\rightarrow$ `v1/configs/legacy/models.yaml` へ移動。
  - `v2/src/likelihood.py`, `v3/src/batch_likelihood.py`, `v3/src/model_utils.py`, `v3/src/ot_utils.py` を legacy へ隔離。
  - root の `plots/` 内の旧図を `docs/archive/plots/` へ移動。

### 3. 検証と最終報告
- [ ] `pytest -q` で全単体テスト通過を確認。
- [ ] `bash scripts/run_production_all.sh cpu --dry-run` を実行し、高速かつエラーなく完走することを確認。
- [ ] `docs/pre_rerun_e6_and_v2_v1_refinements/walkthrough.md` の作成。
