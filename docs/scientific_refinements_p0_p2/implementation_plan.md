# 実装計画: 本実験前 科学的妥当性・P0/P1/P2不整合修正 (全22項目)

論文ストーリーの一貫性：
$$ \text{Covariation} \rightarrow \text{Representation / Causal Overlap} \rightarrow \text{Post-training-Associated Reorganization} \rightarrow \text{Causal Leverage} $$
を維持し、新しいRQは増やさず、既存の証拠を同一の厳密な定義・統計枠組みで計測できるように配線および統計的推論を修正します。

---

## 修正対象コンポーネント一覧と対応方針

### 1. 【P0】V3 Subspace SVD rank-aware 化 & 呼び出し元アンパック修正
- **対象**: `src/affective_empathy_eval/interventions.py`, `v3/primary/run_rq1_state_induction.py`, `v3/primary/run_rq3_path_mediation.py`
- **内容**:
  - `compute_orthonormal_subspace(*directions)` を SVD ベースに改修し、共線に近い軸を SVD 特異値しきい値 $S_i > \text{tol}$ で安全に刈り込む。
  - `run_rq1_state_induction.py` と `run_rq3_path_mediation.py` で `Q_sub, _ = compute_orthonormal_subspace(...)` と tuple アンパックを修正。

### 2. 【P0】V3 `generate_control_directions` API 呼び出し修正
- **対象**: `v3/primary/run_rq1_state_induction.py`
- **内容**:
  - `generate_control_directions(d_v, seed=42)` と `generate_control_directions(d_a, seed=43)` を呼び出し、Valence / Arousal それぞれについて独立した `d_rand` および `d_perp` を生成する。

### 3. 【P0】V3 RQ1 Gate の Valence / Arousal 完全軸別化
- **対象**: `v3/primary/run_rq1_state_induction.py`
- **内容**:
  - Specificity, Necessity, Topic Selectivity (TVD), Gate Decision を VA 両軸で独立に計算。
  - 出力指標に `specificity_v`, `specificity_a`, `attenuation_ratio_v`, `attenuation_ratio_a`, `topic_tvd_v`, `topic_tvd_a` (各 CI 含む) を追加。
  - `decision_v` および `decision_a` を独立に判定。

### 4, 5, 6, 7, 8. 【P0/P1】Behavioral 統計検定・FDR・未定義値・CI の適正化
- **対象**: `behavioral/analysis/summarize_behavioral_aipsy.py`
- **内容**:
  - **4**: RQ1 の Primary 統計量を `aligned_p_value` に統一。Primary FDR family は `task in ['r', 's']` かつ `dimension in ['V', 'A']` に限定。Writer / Dominance は Supplementary に分離。
  - **5**: RQ3 の `displacement_p_value` を Primary とし、存在しない `p_value` 参照エラー (`KeyError`) を解消。
  - **6**: RQ2 の Dose-response について、2段階の単調性検定を Intersection-Union Test（IUT: $p_{\text{monotonic}} = \max(p_{\text{step1}}, p_{\text{step2}})$）で実施し、これに FDR を適用。
  - **7**: `expected direction` 未定義（`n_defined <= 2`）時に raw effect にフォールバックせず `np.nan` を出力。
  - **8**: RQ2 の `slope_ci` で異なる統計量の CI 端点平均を廃止し、Secondary slope $(C - N) / 2$ に対して直接 bootstrap CI を計算。

### 9, 10, 11, 12. 【P0/P1】V2 RQ3 Cross-Fitting & Confirmatory 安全化・Runner接続
- **対象**: `v2/primary/run_rq3_causal_map.py`, `v2/primary/run_confirmatory_analysis.py`, `src/affective_empathy_eval/run.py`
- **内容**:
  - **9**: V2 RQ3 において 5-fold CV を導入し、train fold で $d_V, d_A$ を推定、test fold で介入効果を測定。Base / Instruct, Reader / Self で同一 fold 分割を保証。
  - **10**: 実機実行（非 dry-run）時に成果物がない場合、合成 mock データにフォールバックせず直ちに `FileNotFoundError` / `RuntimeError` を送出。
  - **11**: `run_confirmatory_analysis.py` において H1 (peak / center-of-mass), H2 (Reader–Self sharing), H3 (causal profile reorganization), H4 (distribution recovery) をすべて実装・集約。表現は "post-training-associated reorganization" に厳守。
  - **12**: `src/affective_empathy_eval/run.py` の `run_v2()` 末尾に `run_confirmatory_analysis.py` 実行を追加（単一モデル `--family` 時の条件分岐付き）。

### 13, 14. 【P0/P1】Dry-run / Cache 分離 & Manifest 照合の実接続
- **対象**: V2/V3 各実行スクリプト (`run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`, `run_rq2_spatiotemporal_maps.py`, `run_confirmatory_replication.py`), `src/affective_empathy_eval/manifests.py`
- **内容**:
  - `results/dry_run/` と `results/raw/` の保存ディレクトリを分離。
  - 全スクリプトで `is_manifest_matching(manifest_path, ...)` のファイルパス引数呼び出しに統一し、dry-run キャッシュ誤読を防止。

### 15, 16, 17, 18, 19. 【P1】V3 Confirmatory 統計・評価の精密化
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **内容**:
  - **15**: H1 (Dissociation) および H3 (Endogenous relevance) を Valence / Arousal 両軸に拡張。
  - **16**: Attenuation ratio の `[0, 1]` clip を撤廃し、raw ratio と bootstrap CI を報告。
  - **17**: 論文主結果をバイナリ判定ではなく、効果量推定値および bootstrap 95% CI 中心に構成。
  - **18**: Test fold から常に先頭5件を取る実装を廃止し、乱数シードに基づく再現可能な抽出（pair_id の事前固定）に変更。
  - **19**: H4 の結論を「decodability $\neq$ uniform causal leverage」とし、ステージ特異的なレバレッジ集中を評価。

### 20, 21, 22. 【P2】環境設定・ドキュメント・論文ストーリー表現
- **対象**: `pyproject.toml`, `README.md`, `v2/README.md`
- **内容**:
  - **20**: `pyproject.toml` に `[tool.pytest.ini_options] pythonpath = ["src", "."]` を追加。
  - **21**: ルート `README.md` の重複見出し（`## 1. 中心リサーチクエスチョン`）を削除。
  - **22**: V2 ドキュメントにおいて、「固定モデル内の介入効果（因果）」と「Base–Instruct 比較（post-training-associated な観察的帰属）」を明確に区別して記述。

---

## 検証計画
1. **静的検証・テスト実行**:
   - `python -m pytest -q` を実行し、全既存テストおよび新規テストの通過を確認（`PYTHONPATH` を明示指定せず直接実行）。
2. **スクリプト構文・インポート検証**:
   - `python -m compileall src behavioral v1 v2 v3 tests`
3. **ドライラン検証**:
   - Behavioral, V2, V3 の各分析スクリプトを `--dry-run` で実行し、キャッシュ分離、manifest 生成、エラーなく完了することを確認。
