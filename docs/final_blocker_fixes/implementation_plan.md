# 最終 Blocker 修正・本番即応化計画 (Implementation Plan)

## 概要
ユーザーレビューで指摘された 5 大 BLOCKER（V1 Phase C の `yaml` import 欠落、Behavioral AIPsy 集計のファイル名不一致、AIPsy dry-run の本番 schema 非再現、Sequence-Likelihood の length-normalized 定義・実装不一致、古いキャッシュ再利用の防止）および統計・再現性関連の要修正事項を修正し、全 Stage の dry-run と pytest を完走させ、本番実行可能な状態を確立する。

---

## 修正項目と実装方針

### 1. BLOCKER 1: V1 Phase C `import yaml` 欠落
- `v1/primary/run_phase_c.py` の import 部分に `import yaml` を追加。

### 2. BLOCKER 2: Behavioral AIPsy 集計のファイル名照合と例外化
- `behavioral/analysis/summarize_behavioral_aipsy.py`:
  - `glob.glob` の対象を canonical な `behavioral_aipsy_*_4split.csv` に修正（重複集計の防止）。
  - マッチするファイルが 0 件の場合は単なる `return` ではなく `FileNotFoundError` を送出。
  - モデル名抽出を `stem` から `behavioral_aipsy_` と `_4split` を除去した形式（例: `qwen_base`, `qwen_instruct`）に修正。

### 3. BLOCKER 3: AIPsy dry-run の本番 schema 化
- `behavioral/primary/run_behavioral_aipsy.py`:
  - dry-run mock 生成時に `res_df = pd.read_csv(stim_path).copy()` をベースとし、`split`, `pair_id`, `triplet_id`, `emotion` 等のスキーマを完全に保持。
  - `w_ev`, `w_ea`, `w_ed`, `r_ev`, `r_ea`, `r_ed`, `s_ev`, `s_ea`, `s_ed` を deterministic に付与。
  - 単体テストを追加し、集計後に 4 つの Derived ファイル（`behavioral_aipsy_sensitivity_rq1.csv`, `behavioral_aipsy_dose_response_rq2.csv`, `behavioral_aipsy_specificity_rq3.csv`, `behavioral_aipsy_coupling_rq4.csv`）が実在することを確認。

### 4. BLOCKER 4: Sequence-Likelihood の length-normalized 統一
- 論文定義 $s_y(x) = \frac{1}{|y|}\sum_t \log p(y_t \mid x, y_{<t})$ に整合させる。
- `src/affective_empathy_eval/likelihood.py`:
  - `compute_sequence_likelihoods_for_candidates` のデフォルト引数を `normalize_length: bool = True` に変更。
- Behavioral, V1, V2, V3 の Primary Sequence-Likelihood 呼び出しをすべて `normalize_length=True` に統一。
- 各 YAML 設定ファイル（`configs/v1_experiments.yaml`, `v2_experiments.yaml`, `v3_experiments.yaml` 等）に以下を明示：
  ```yaml
  sequence_likelihood:
    normalize_length: true
    temperature: 1.0
  ```
- manifest に `"sequence_likelihood_normalization": "token_mean"`, `"temperature": 1.0` を記録。

### 5. BLOCKER 5: キャッシュ無効化 (Cache Invalidation) の厳密化
- `affective_empathy_eval/manifests.py`:
  - `DEFAULT_CODE_VERSION` を `"2.2.0"` に更新。
  - `sequence_likelihood_normalization` や scoring 方式を manifest の比較対象に含める。
  - `is_manifest_matching` に `expected_code_version` を接続。

### 6. 統計修正: `compute_d_z()` のゼロ分散処理
- `src/affective_empathy_eval/statistics.py`:
  - `s_delta < 1e-9` のとき `0.0` ではなく `np.nan` を返すよう修正。

### 7. 定義明文化: FDR Family & AIPsy Direction Map
- `behavioral/README.md` および Methods に FDR 検定 family（Sensitivity, Dose-response, Specificity, Coupling）を明記。
- `affect_directions.py` の hash / version を manifest に保存。

---

## 検証ステップ
1. コード修正の適用
2. 単体テスト実行 (`pytest -q`)
3. 全 Stage の dry-run 実行確認
   - Behavioral dry-run $\rightarrow$ derived 4ファイル生成確認
   - V1 Phase C Base/Instruct dry-run $\rightarrow$ summary 確認
   - V2 dry-run $\rightarrow$ 確認
   - V3 dry-run $\rightarrow$ gate $\rightarrow$ RQ2/3/confirmatory 確認
4. `walkthrough.md` の作成と報告
