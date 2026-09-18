# 実装計画: 全再実行直前 E6 Discovery/Confirmation 分離・V2 整合化・キャッシュ厳格化

## 概要
全再実行開始直前の最終リファインメントとして、以下を実施します：
1. **V1 E6**: E3 Discovery site 選択 $\rightarrow$ Confirmation pairs のみでの targeted ablation 検定への厳格分離、および用語の「causal specialization / partial dissociation」への統一。
2. **V2 RQ2**: `delta_delta_cross` の計算式を $(I_{R \to S} - B_{R \to S}) - (I_{S \to R} - B_{S \to R})$ に修正。
3. **V2 dry-run**: サンプル数 $N \le 32$、`hidden_dim <= 64`、`layers <= 4` に縮小し、数秒で完走する軽量スモークテスト化。
4. **V1 Phase C cache**: `model_id`, `git_commit`, `dataset_hash`, `prompt_hash`, `candidate_schema`, `intervention_position`, `tokenizer` のメタデータハッシュ検証を導入。
5. **V2 Procrustes**: rank-deficiency 回避のため、PCA（$k=64$）による次元削減後の Procrustes アラインメントへ改修。
6. **リポジトリ整理**: `v1/configs/models.yaml`、未使用の `v2/src/`, `v3/src/`、root `plots/` を `legacy/` や `docs/archive/` へ移動。

---

## ユーザーレビュー必須項目

> [!IMPORTANT]
> **V1 E6 の実験分離（Discovery vs Confirmation）**:
> - E6 実行時、データセットを `pair_id` に基づいて厳格に 50:50 の Discovery split と Confirmation split に分割します。
> - Discovery split（または先行する E3 Discovery 解析結果）から Reader ピーク層 $l^*_R$ および Self ピーク層 $l^*_S$ を動的に決定します。
> - 決定したターゲット層に対する 2x2 ablation（$l^*_R \times l^*_S$）は、完全に独立した **Confirmation split のみ** を用いて検定します。

> [!IMPORTANT]
> **V2 dry-run の軽量化**:
> - これまで full dataset × full hidden_dim × 全層（28〜32層）の重い擬似活性化テンソルを構築し SVD を走らせていたのを、`N=32`, `hidden_dim=64`, `layers=4` に縮小し、結合テストが即座に終わるようにします。

---

## 提案される変更点

### 1. V1 Phase C E6 Specialization (`v1/primary/phase_c/run_e6_specialization.py`)
- 50/50 Group split (`pair_id` に基づく分割) を導入。
- Discovery 側で各層の介入応答（$\Delta \mathrm{EV}_{\mathrm{reader}}, \Delta \mathrm{EV}_{\mathrm{self}}$）を評価し、$l^*_R = \arg\max \Delta \mathrm{EV}_{\mathrm{reader}}$, $l^*_S = \arg\max \Delta \mathrm{EV}_{\mathrm{self}}$ を同定。
- Confirmation pairs のみで 2x2 ablation（baseline, abl_reader, abl_self, abl_both）を実行し、LMM または paired 差分を算出。
- 表示・ファイル名・ログで「Double Dissociation」を「Causal Specialization / Partial Dissociation」に表現緩和。

### 2. V2 Cross-decoding 指標修正 (`v2/primary/run_rq1_rq2_cross_decoding.py`)
- `delta_delta_cross` を以下のように修正：
  $$\Delta\Delta_{\mathrm{cross}} = (I_{R \to S} - B_{R \to S}) - (I_{S \to R} - B_{S \to R})$$
  （Instruct vs Base における、Reader $\to$ Self 方向の変化と Self $\to$ Reader 方向の変化の差）

### 3. V2 dry-run 軽量化 & Procrustes PCA
- `v2/primary/run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`:
  - `args.dry_run` 時は $N=32$, `num_layers=4`, `hidden_dim=64` を使用。
  - Procrustes 計算時：高次元特異性を回避するため、train split で fit した PCA（$k=64$）を適用後に直交回転行列を算出。

### 4. V1 Phase C キャッシュハッシュ検証 (`v1/primary/run_phase_c.py`)
- キャッシュ保存時に `cache_metadata.json` を書き出し（`model_id`, `git_commit`, `dataset_sha256`, `prompt_hash`, `intervention_position` 等）。
- ロード時にハッシュが完全一致しない場合はキャッシュを無効化し、自動再計算。

### 5. 古い設定・未使用コード・成果物の整理
- `v1/configs/models.yaml` $\rightarrow$ `v1/configs/legacy/models.yaml`
- `v2/src/likelihood.py` $\rightarrow$ `v2/src/legacy/`
- `v3/src/batch_likelihood.py`, `v3/src/model_utils.py`, `v3/src/ot_utils.py` $\rightarrow$ `v3/src/legacy/`
- `plots/activation_patching_plot.png`, `output_gating_plot.png` $\rightarrow$ `docs/archive/plots/`

---

## 検証計画
- 単体テスト: `.venv/bin/pytest -q`（全件通過）
- 全ステージ dry-run テスト: `bash scripts/run_production_all.sh cpu --dry-run`（数分以内で高速完走確認）
- キャッシュ整合性テスト: パラメータ変更時のキャッシュ無効化確認
